import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import cv2
import time
import numpy as np

from networks.ReinitNet import GlobalReinitNet, LocalReinitNet
from networks.SmallMobileNet import SmallMobileNetV2, SmallMobileNetV2Part

class DPRGNet(nn.Module):

    def __init__(self):
        super(DPRGNet, self).__init__()
		
    def load_model(self, args):
	
        ## 1. Load the global reinitialization model
        self.global_reinit_net = GlobalReinitNet()
        global_reinit_checkpoint = torch.load(args.global_reinit_path, map_location=lambda storage, loc: storage)['state_dict']
        global_reinit_model_dict = self.global_reinit_net.state_dict()
		
        # Because the model is trained by multiple gpus, prefix module should be removed.
        for k in global_reinit_checkpoint.keys():
            global_reinit_model_dict[k.replace('module.', '')] = global_reinit_checkpoint[k]
        self.global_reinit_net.load_state_dict(global_reinit_model_dict, strict=False)
		
        # 2. Load the global regression model
        self.global_regress_net = SmallMobileNetV2(num_classes = args.pts_num * 2)
        global_regress_checkpoint = torch.load(args.global_regress_path, map_location=lambda storage, loc: storage)['state_dict']
        global_regress_model_dict = self.global_regress_net.state_dict()
		
        for k in global_regress_checkpoint.keys():
            global_regress_model_dict[k.replace('module.', '')] = global_regress_checkpoint[k]
        self.global_regress_net.load_state_dict(global_regress_model_dict)

        self.mode = args.mode
        if self.mode == 'gpu':
            self.device = torch.device("cuda:%d"%(args.devices_id))
            cudnn.benchmark = True
            self.global_reinit_net = self.global_reinit_net.to(self.device)
            self.global_regress_net = self.global_regress_net.to(self.device)
        self.global_reinit_net.eval()
        self.global_regress_net.eval()
		
        # 3. load the auxiliary model and operation
        self.pts_num = args.pts_num
        self.mean_face = np.load(args.global_mean_face_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
        self.reinit_input_size = args.global_reinit_size
        self.regress_input_size = args.global_regress_size
        self.dst_pts = np.float32([[0, 0], [0, self.reinit_input_size - 1], [self.reinit_input_size - 1, 0]])


    def predict(self, ori_img, face_rect):
	
        # 1. Get the input for global reinitialization
        src_pts = np.float32([[face_rect[0], face_rect[1]], [face_rect[0], face_rect[3]], [face_rect[2], face_rect[1]]])
        tform0 = cv2.getAffineTransform(src_pts, self.dst_pts)
        img = cv2.warpAffine(ori_img, tform0, (self.reinit_input_size, self.reinit_input_size))

        # 2.Global reinitialization inference
        input = self.transform(img).unsqueeze(0)
        start = time.time()
        with torch.no_grad():
            if self.mode == 'gpu':
                input = input.to(self.device)
            greinit_param = self.global_reinit_net(input)
            greinit_param = greinit_param[0].squeeze().cpu().numpy().flatten().astype(np.float32)
        end = time.time()
        print( 'Global reinit infer: %f ms\n'%(1000*(end-start)))

        # 3. Get the normarlized state
        tform0 = np.vstack((tform0, [0, 0, 1]))
        tform1 = greinit_param.reshape((2, 3))
        tform1[0:2,2] = tform1[0:2,2] * self.reinit_input_size
        tform1 = np.vstack((tform1, [0, 0, 1]))
        s_tform = [[self.regress_input_size/self.reinit_input_size,0,0],[0,self.regress_input_size/self.reinit_input_size,0],[0,0,1]]
        tform = np.dot(tform1, tform0)
        tform = np.dot(s_tform, tform)
        inv_tform = np.linalg.inv(tform)
        tform = tform[0:2,:]
        inv_tform = inv_tform[0:2,:]
        new_img = cv2.warpAffine(ori_img, tform, (self.regress_input_size, self.regress_input_size))

        # 4. Global regression inference
        input = self.transform(new_img).unsqueeze(0)
        start = time.time()
        with torch.no_grad():
            if self.mode == 'gpu':
                input = input.to(self.device)
            gregress_param = self.global_regress_net(input)
            gregress_param = gregress_param[0].squeeze().cpu().numpy().flatten().astype(np.float32)
        end = time.time()
        print("Global pts net infere: %f ms\n"%(1000*(end-start)))
        
        # 5. Global shape projection
        pre_pts = []
        for i in range(self.pts_num):
           tmp_x = (gregress_param[2*i]*0.5+self.mean_face[2*i])*self.regress_input_size
           tmp_y = (gregress_param[2*i+1]*0.5+self.mean_face[2*i+1])*self.regress_input_size
           vec = np.array([[tmp_x],[tmp_y],[1]])
           pts = np.dot(inv_tform,vec)
           pts = np.squeeze(pts)
           pre_pts.append(pts)

        return pre_pts


class DPRGLNet(nn.Module):
    def __init__(self):
        super(DPRGLNet, self).__init__()

    def load_model(self, args):

        # 1. Load the global reinitialization model
        self.global_reinit_net = GlobalReinitNet()
        global_reinit_checkpoint = torch.load(args.global_reinit_path, map_location=lambda storage, loc: storage)[
            'state_dict']
        global_reinit_model_dict = self.global_reinit_net.state_dict()

        for k in global_reinit_checkpoint.keys():
            global_reinit_model_dict[k.replace('module.', '')] = global_reinit_checkpoint[k]
        self.global_reinit_net.load_state_dict(global_reinit_model_dict, strict=False)

        # 2. Load the global regression model
        self.global_regress_net = SmallMobileNetV2(num_classes=args.pts_num * 2)
        global_regress_checkpoint = torch.load(args.global_regress_path, map_location=lambda storage, loc: storage)[
            'state_dict']
        global_regress_model_dict = self.global_regress_net.state_dict()

        for k in global_regress_checkpoint.keys():
            global_regress_model_dict[k.replace('module.', '')] = global_regress_checkpoint[k]
        self.global_regress_net.load_state_dict(global_regress_model_dict)

        # 3. Load the local reinitialization model
        self.local_reinit_net = LocalReinitNet(args.local_reinit_size)
        local_reinit_checkpoint = torch.load(args.local_reinit_path, map_location=lambda storage, loc: storage)[
            'state_dict']
        local_reinit_model_dict = self.local_reinit_net.state_dict()
        for k in local_reinit_checkpoint.keys():
            local_reinit_model_dict[k.replace('module.', '')] = local_reinit_checkpoint[k]
        self.local_reinit_net.load_state_dict(local_reinit_model_dict)

        # 4. Load the local regression model
        self.local_regress_net = SmallMobileNetV2Part(num_classes=args.pts_num * 2)
        local_regress_checkpoint = torch.load(args.local_regress_path, map_location=lambda storage, loc: storage)[
            'state_dict']
        local_regress_model_dict = self.local_regress_net.state_dict()
        for k in local_regress_checkpoint.keys():
            local_regress_model_dict[k.replace('module.', '')] = local_regress_checkpoint[k]
        self.local_regress_net.load_state_dict(local_regress_model_dict)

        self.mode = args.mode
        if self.mode == 'gpu':
            self.device = torch.device("cuda:%d" % (args.devices_id))
            cudnn.benchmark = True
            self.global_reinit_net = self.global_reinit_net.to(self.device)
            self.global_regress_net = self.global_regress_net.to(self.device)
            self.local_reinit_net = self.local_reinit_net.to(self.device)
            self.local_regress_net = self.local_regress_net.to(self.device)
        self.global_reinit_net.eval()
        self.global_regress_net.eval()
        self.local_reinit_net.eval()
        self.local_regress_net.eval()

        # 5. Load the auxiliary model and operation
        self.pts_num = args.pts_num
        self.mean_face = np.load(args.global_mean_face_path)
        data = np.load(args.local_mean_face_path, allow_pickle=True)
        self.mean_left_eye = data[0]
        self.mean_right_eye = data[1]
        self.mean_nose = data[2]
        self.mean_mouth = data[3]
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
        self.global_reinit_input_size = args.global_reinit_size
        self.global_regress_input_size = args.global_regress_size
        self.local_regress_input_size = args.local_regress_size
        self.dst_pts = np.float32([[0, 0], [0, self.global_reinit_input_size - 1], [self.global_reinit_input_size - 1, 0]])
        self.left_eye_idx = args.left_eye_idx
        self.right_eye_idx = args.right_eye_idx
        self.nose_idx = args.nose_idx
        self.mouth_idx = args.mouth_idx

    def predict(self, ori_img, face_rect):

        # 1. Get the input for global reinitialization
        src_pts = np.float32([[face_rect[0], face_rect[1]], [face_rect[0], face_rect[3]], [face_rect[2], face_rect[1]]])
        tform0 = cv2.getAffineTransform(src_pts, self.dst_pts)
        img = cv2.warpAffine(ori_img, tform0, (self.global_reinit_input_size, self.global_reinit_input_size))

        # 2. Global reinitialization inference
        input = self.transform(img).unsqueeze(0)
        start = time.time()
        with torch.no_grad():
            if self.mode == 'gpu':
                input = input.to(self.device)
            greinit_param = self.global_reinit_net(input)
            greinit_param = greinit_param[0].squeeze().cpu().numpy().flatten().astype(np.float32)
        end = time.time()
        print('Global reinit infer: %f ms\n' % (1000 * (end - start)))

        # 3. Get the normarlized state
        tform0 = np.vstack((tform0, [0, 0, 1]))
        tform1 = greinit_param.reshape((2, 3))
        tform1[0:2, 2] = tform1[0:2, 2] * self.global_reinit_input_size
        tform1 = np.vstack((tform1, [0, 0, 1]))
        s_tform = [[self.global_regress_input_size / self.global_reinit_input_size, 0, 0],
                   [0, self.global_regress_input_size / self.global_reinit_input_size, 0], [0, 0, 1]]
        tform = np.dot(tform1, tform0)
        tform = np.dot(s_tform, tform)
        inv_tform = np.linalg.inv(tform)
        tform = tform[0:2, :]
        inv_tform = inv_tform[0:2, :]
        new_img = cv2.warpAffine(ori_img, tform, (self.global_regress_input_size, self.global_regress_input_size))

        # 4. Global regression inference
        input = self.transform(new_img).unsqueeze(0)
        start = time.time()
        with torch.no_grad():
            if self.mode == 'gpu':
                input = input.to(self.device)
            gregress_param = self.global_regress_net(input)
            gregress_param = gregress_param[0].squeeze().cpu().numpy().flatten().astype(np.float32)
        end = time.time()
        print("Global pts net infere: %f ms\n" % (1000 * (end - start)))

        # 5. Global shape projection
        pre_pts = []
        gcrop_pts = []
        for i in range(self.pts_num):
            tmp_x = (gregress_param[2 * i] * 0.5 + self.mean_face[2 * i]) * self.global_regress_input_size
            tmp_y = (gregress_param[2 * i + 1] * 0.5 + self.mean_face[2 * i + 1]) * self.global_regress_input_size
            vec = np.array([[tmp_x], [tmp_y], [1]])
            pts = np.dot(inv_tform, vec)
            pts = np.squeeze(pts)
            pre_pts.append(pts)
            gcrop_pts.append([tmp_x, tmp_y])
        pre_pts = np.array(pre_pts)
        gcrop_pts = np.array(gcrop_pts)

        # 6. Local reinitialization inference
        input = gcrop_pts.ravel()
        input = input.astype(np.float32)
        input = torch.from_numpy(input)
        start = time.time()
        with torch.no_grad():
            if self.mode == 'gpu':
                input = input.to(self.device)
            lreinit_param = self.local_reinit_net(input)
            lreinit_param = [x.cpu().numpy().astype(np.float32) for x in lreinit_param]
        end = time.time()
        print('Local stn infere: %f ms\n' % (1000 * (end - start)))

        left_eye_tform = lreinit_param[0].reshape((2, 3))
        left_eye_tform[0:2, 2] = left_eye_tform[0:2, 2] * self.local_regress_input_size

        right_eye_tform = lreinit_param[1].reshape((2, 3))
        right_eye_tform[0:2, 2] = right_eye_tform[0:2, 2] * self.local_regress_input_size

        nose_tform = lreinit_param[2].reshape((2, 3))
        nose_tform[0:2, 2] = nose_tform[0:2, 2] * self.local_regress_input_size

        mouth_tform = lreinit_param[3].reshape((2, 3))
        mouth_tform[0:2, 2] = mouth_tform[0:2, 2] * self.local_regress_input_size

        left_eye_inv_tform = np.linalg.inv(np.vstack((left_eye_tform, [0, 0, 1])))
        left_eye_inv_tform = left_eye_inv_tform[0:2, :]
        left_eye_img = cv2.warpAffine(new_img, left_eye_tform, (self.local_regress_input_size,
                                                                self.local_regress_input_size))

        right_eye_inv_tform = np.linalg.inv(np.vstack((right_eye_tform, [0, 0, 1])))
        right_eye_inv_tform = right_eye_inv_tform[0:2, :]
        right_eye_img = cv2.warpAffine(new_img, right_eye_tform, (self.local_regress_input_size,
                                                                  self.local_regress_input_size))

        nose_inv_tform = np.linalg.inv(np.vstack((nose_tform, [0, 0, 1])))
        nose_inv_tform = nose_inv_tform[0:2, :]
        nose_img = cv2.warpAffine(new_img, nose_tform, (self.local_regress_input_size,
                                                        self.local_regress_input_size))

        mouth_inv_tform = np.linalg.inv(np.vstack((mouth_tform, [0, 0, 1])))
        mouth_inv_tform = mouth_inv_tform[0:2, :]
        mouth_img = cv2.warpAffine(new_img, mouth_tform, (self.local_regress_input_size,
                                                          self.local_regress_input_size))

        '''
        cv2.imwrite('new_img.jpg', new_img) 
        cv2.imwrite('left_eye_img.jpg', left_eye_img)
        cv2.imwrite('right_eye_img.jpg', right_eye_img)
        cv2.imwrite('nose_img.jpg', nose_img)
        cv2.imwrite('mouth_img.jpg', mouth_img)
        pdb.set_trace()
        '''
        # 7. local regression inference
        input_left_eye = self.transform(left_eye_img).unsqueeze(0)
        input_right_eye = self.transform(right_eye_img).unsqueeze(0)
        input_nose = self.transform(nose_img).unsqueeze(0)
        input_mouth = self.transform(mouth_img).unsqueeze(0)

        input = torch.stack([input_left_eye, input_right_eye, input_nose, input_mouth])

        start = time.time()
        with torch.no_grad():
            if self.mode == 'gpu':
                input = input.to(self.device)
            params = self.local_regress_net(input)
        end = time.time()
        print("Local pts net infere: %f ms\n" % (1000 * (end - start)))
		
		
		# 8. Local shape projection
        param = params[0].squeeze().cpu().numpy().flatten().astype(np.float32)
        for i in range(len(self.mean_left_eye)):
            tmp_x = (param[2 * i] * 0.5 + self.mean_left_eye[i][0]) * self.local_regress_input_size
            tmp_y = (param[2 * i + 1] * 0.5 + self.mean_left_eye[i][1]) * self.local_regress_input_size
            vec = np.array([[tmp_x], [tmp_y], [1]])
            pts = np.dot(left_eye_inv_tform, vec)
            pts = np.squeeze(pts)
            vec = np.array([[pts[0]], [pts[1]], [1]])
            pts = np.dot(inv_tform, vec)
            pts = np.squeeze(pts)
            pre_pts[self.left_eye_idx[i]] = pts

        param = params[1].squeeze().cpu().numpy().flatten().astype(np.float32)
        for i in range(len(self.mean_right_eye)):
            tmp_x = (param[2 * i] * 0.5 + self.mean_right_eye[i][0]) * self.local_regress_input_size
            tmp_y = (param[2 * i + 1] * 0.5 + self.mean_right_eye[i][1]) * self.local_regress_input_size
            vec = np.array([[tmp_x], [tmp_y], [1]])
            pts = np.dot(right_eye_inv_tform, vec)
            pts = np.squeeze(pts)
            vec = np.array([[pts[0]], [pts[1]], [1]])
            pts = np.dot(inv_tform, vec)
            pts = np.squeeze(pts)
            pre_pts[self.right_eye_idx[i]] = pts

        param = params[2].squeeze().cpu().numpy().flatten().astype(np.float32)
        for i in range(len(self.mean_nose)):
            tmp_x = (param[2 * i] * 0.5 + self.mean_nose[i][0]) * self.local_regress_input_size
            tmp_y = (param[2 * i + 1] * 0.5 + self.mean_nose[i][1]) * self.local_regress_input_size
            vec = np.array([[tmp_x], [tmp_y], [1]])
            pts = np.dot(nose_inv_tform, vec)
            pts = np.squeeze(pts)
            vec = np.array([[pts[0]], [pts[1]], [1]])
            pts = np.dot(inv_tform, vec)
            pts = np.squeeze(pts)
            pre_pts[self.nose_idx[i]] = pts

        param = params[3].squeeze().cpu().numpy().flatten().astype(np.float32)
        for i in range(len(self.mean_mouth)):
            tmp_x = (param[2 * i] * 0.5 + self.mean_mouth[i][0]) * self.local_regress_input_size
            tmp_y = (param[2 * i + 1] * 0.5 + self.mean_mouth[i][1]) * self.local_regress_input_size
            vec = np.array([[tmp_x], [tmp_y], [1]])
            pts = np.dot(mouth_inv_tform, vec)
            pts = np.squeeze(pts)
            vec = np.array([[pts[0]], [pts[1]], [1]])
            pts = np.dot(inv_tform, vec)
            pts = np.squeeze(pts)
            pre_pts[self.mouth_idx[i]] = pts

        pre_pts = np.array(pre_pts)

        return pre_pts
