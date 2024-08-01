import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pil_im
from PIL import ImageFont, ImageDraw, ImageEnhance
import  scipy.io
import gdown

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.io import read_image

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from IPython.display import Image

# @title get image

class SquarePad:
    def __call__(self, image):
        s = image.shape
        max_wh = max(s[-1], s[-2])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)
        padding = (hp, hp, vp, vp)
        return F.pad(image, padding, 'constant', 0)

class norm_to_zo:
    def __call__(self, image):
        return image/255.0

class FeatureExtractorNet:
    def __init__(self, model_name='resnet18', device=torch.device('cpu')):
        self.model_name = model_name
        self.device = device
        self.model = self.get_model()

    def get_model(self):
        if self.model_name == 'resnet18':
            model_conv = models.resnet18(pretrained=True)
        # Add more models as needed, e.g., elif self.model_name == 'resnet50': ...
        else:
            raise ValueError("Invalid model name.")

        for param in model_conv.parameters():
            param.requires_grad = False

        model_feature = torch.nn.Sequential(*(list(model_conv.children())[:-1]))
        return model_feature.to(self.device)

    def extract_features(self, img):
        self.model.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            SquarePad(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            features = self.model(img)
        return features

class ObjectDetectorYOLO:
    def __init__(self, model_name='yolov5s'):
        self.model_name = model_name
        self.model = torch.hub.load('ultralytics/yolov5', self.model_name, pretrained=True)

    def detect_objects(self, image_path):
        results = self.model(image_path)
        return results.crop(save=False)


class AffordanceAnalyzer:
    def __init__(self, model_name='resnet18', dataset_name='RGBDAffordance', img_size=224, device=torch.device('cpu'), nr_of_bases=20, auto_threshold = 0.8, plot_graph=False):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.device = device
        self.nr_of_bases = nr_of_bases
        self.auto_threshold = auto_threshold
        self.plot_graph = plot_graph
        self.object_labels = ['ball', 'binder', 'bowl', 'calculator', 'camera','cap', 'cell_phone',
                'cereal_box', 'coffee_mug','comb', 'dry_battery', 'flashlight', 'food_bag',
                'food_box', 'food_can', 'food_cup', 'food_jar', 'glue_stick', 'hand_towel',
                'instant_noodles', 'keyboard','kleenex', 'lightbulb', 'marker', 'notebook',
                'pitcher', 'plate', 'pliers', 'rubber_eraser', 'scissors', 'shampoo',
                 'soda_can', 'sponge', 'stapler', 'toothbrush', 'toothpaste', 'water_bottle']
        self.affordance_names_T = ['Grasp', 'Wrap Grasp', 'Contain', 'Open', 'Tip-push', 'Display',
                    'Roll', 'Dry', 'Liquid contain', 'Pour', 'Grip', 'Absorb',
                    'Cut', 'Staple', 'Illuminate']
        self.affordance_names = ['grasp', 'wrap grasp', 'containment', 'openable', 'tip-push', 'display',
                    'rollable', 'dry', 'liquid_containment', 'pourable', 'grip', 'absorb',
                    'cut', 'stapling', 'illumination']

        self.model_name_T = 'ResNet18' if model_name == 'resnet18' else 'RegNetY'


        w_matr =np.loadtxt(r'/content/W_matr_%s.csv'%self.model_name, delimiter=',')
        w_max = np.max(w_matr)
        self.W_matr = torch.tensor(w_matr/w_max)

        self.base_list = dict()
        self.base_point_vecs = dict()
        self.state_dict = dict()
        self.threshold_dict = dict()
        self.afford_labellist = list()
        self.afford_dict = dict()
        self.afford_dict_T = dict()


    def download_files(self):
        try:
            os.mkdir(r'/content/%s'%self.model_name)
        except:
            pass
        if self.model_name == 'resnet18':
            gdown.download(id = '10rQgt720xNfqwaKkAZC8-DoHyqPVjHAQ')
            gdown.download(id = '10rsMnDM6LAd1DdOsuRAwoJ-oz8wasymL')
            gdown.download(id = '10mlW-u062prfC-R7RxlQKPuf-aSN6cgM')
            gdown.download(id = '1VYW0W9mqjzyrBRPlpCbXlw3KZ6pWGZgd')
        
        elif self.model_name == 'regnet_y':
            gdown.download(id = '1-xut_M1T2hpTsGFcKoc8LGsSSVDrwske')
            gdown.download(id = '103CFnryrTUFSZFhmLL6-tP9ui5vPPU4-')
            gdown.download(id = '1-s_Kr3O7uP0KTL91Ss_V8Fyho6s6OWG6')
            gdown.download(id = '1GvxUUPwOaT_fLew4N0z5nAr3GqMydGhw')      

    def take_photo(self, filename='photo.jpg', quality=0.8):
        js = Javascript('''
            async function takePhoto(quality) {
                const div = document.createElement('div');
                const capture = document.createElement('button');
                capture.textContent = 'Capture';
                div.appendChild(capture);

                const video = document.createElement('video');
                video.style.display = 'block';
                const stream = await navigator.mediaDevices.getUserMedia({video: true});

                document.body.appendChild(div);
                div.appendChild(video);
                video.srcObject = stream;
                await video.play();

                // Resize the output to fit the video element.
                google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

                // Wait for Capture to be clicked.
                await new Promise((resolve) => capture.onclick = resolve);

                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                stream.getVideoTracks()[0].stop();
                div.remove();
                return canvas.toDataURL('image/jpeg', quality);
            }
            ''')
        display(js)
        data = eval_js('takePhoto({})'.format(quality))
        binary = b64decode(data.split(',')[1])
        with open(filename, 'wb') as f:
            f.write(binary)

    def get_tens(self):
        # state vectors obtained from Resnet are loaded from previously saved file
        st_tns = torch.load(r'/content/%s_with_%s_%dnetworkout.pt'%(self.model_name, self.dataset_name, self.img_size))
        st_tns = torch.transpose(st_tns, 0, 1)
        # also results of those tensor are loaded. Results contain affordance labels, like 0, 1, 2,
        rslt_tns = torch.load(r'/content/%s_with_%s_%dlabel.pt'%(self.model_name, self.dataset_name, self.img_size))

        # image paths are loaded to name list
        # image names also contain names of the object/s in the image
        nm_list = []
        f = open(r'/content/%s_with_%s_%dimagepaths.txt'%(self.model_name, self.dataset_name, self.img_size), 'r')
        for line in f:
            nm_list= line.strip().split('#')
        f.close()

        return st_tns, rslt_tns, nm_list

    def project_matr(self, bases):
        return torch.matmul(torch.matmul(bases, torch.linalg.inv(
            torch.matmul(torch.transpose(bases, 0, 1), bases) )), torch.transpose(bases, 0, 1))

    def matr_to_origin(self, tens, indx):
        [dim1, dim2] = tens.shape
        t = tens[:, indx]
        t = t.resize(dim1, 1).expand(dim1, dim2)
        return tens-t

    def matr_vec_to_origin(self, tens, vec):
        [dim1, dim2] = tens.shape
        vec = vec.resize(dim1, 1).expand(dim1, dim2)
        return tens-vec

    def matr_zero_mean(self, M):
        [dim1, dim2] = M.shape
        mn = torch.mean(M, 1)
        mn = mn.resize(dim1, 1).expand(dim1, dim2)
        return M-mn

    def calc_mi(self, m, ind,rdim, nnum):
        mto = self.matr_to_origin(m, ind)
        dig = torch.diag(torch.matmul(torch.transpose(mto, 0, 1), mto))
        salad, ind = torch.sort(dig)
        ind_e = ind[1:nnum+1]
        mi= self.matr_zero_mean(m[:, ind_e])
        u, s, vh = torch.linalg.svd(mi, full_matrices=True)
        return( torch.matmul(u[:,:rdim], torch.diag(s[:rdim])), s[:rdim], ind_e)

    def curv_calc(self, m, vct, rdim, nnum):
        mtp = self.matr_vec_to_origin(m, vct)
        dig = torch.diag(torch.matmul(torch.transpose(mtp, 0, 1), mtp))
        salad, ind = torch.sort(dig)
        ind_e = ind[0:nnum]
        ind_w = ind[1:nnum]
        mn = m[:, ind_e]
        mni = self.matr_zero_mean(mn)
        #mw = m[:, ind_w]
        #mwi = matr_zero_mean(mw)
        mtot = torch.concat((mn, vct), 1)
        mtoti = self.matr_zero_mean(mtot)

        un, sn, vn = torch.linalg.svd(mni, full_matrices = True)
        utot, stot, vtot = torch.linalg.svd(mtoti, full_matrices = True)
        #uw, sw, vw = torch.linalg.svd(mwi, full_matrices = True)

        usn = torch.matmul(un[:,:rdim], torch.diag(sn[:rdim]))
        ustot = torch.matmul(utot[:,:rdim], torch.diag(stot[:rdim]))
        #usw = torch.matmul(uw[:,:rdim], torch.diag(sw[:rdim]))

        Q = torch.matmul(torch.transpose(usn, 0, 1), ustot)
        uq, sq, vq = torch.linalg.svd(Q)
        theta = torch.acos(torch.abs(torch.clamp(torch.sum(sq)/torch.sum(sn[:rdim]*stot[:rdim]), min=-1.0, max=1.0)))
        return theta#/theta_w

    def curv_calc_auto(self,m, vct):
        mtp = self.matr_vec_to_origin(m, vct)
        dig = torch.diag(torch.matmul(torch.transpose(mtp, 0, 1), mtp))
        salad, ind = torch.sort(dig)

        neighbour_number_list = list()
        neighbour_dim_list = list()

        for indx_num in range(2,100):
            #print(indx_num)
            ind_e = ind[0:indx_num]
            ind_w = ind[1:indx_num]
            mn = m[:, ind_e]
            mni = self.matr_zero_mean(mn)
            #mw = m[:, ind_w]
            #mwi = matr_zero_mean(mw)
            mtot = torch.concat((mn, vct), 1)
            mtoti = self.matr_zero_mean(mtot)

            un, sn, vn = torch.linalg.svd(mni, full_matrices = True)
            utot, stot, vtot = torch.linalg.svd(mtoti, full_matrices = True)
            #uw, sw, vw = torch.linalg.svd(mwi, full_matrices = True)
            #print(sn)

            energy_tensor = torch.cumsum(sn, dim = 0)/torch.sum(sn, dim = 0)
            #print(energy_tensor)
            #print(energy_tensor)
            try:
                rdim = torch.min((energy_tensor > self.auto_threshold).nonzero().squeeze()).item()
            except:
                rdim = 0
            neighbour_number_list.append(indx_num)
            neighbour_dim_list.append(rdim)
            if 3<=indx_num:
                if rdim<=neighbour_dim_list[-2]:
                    break
        usn = torch.matmul(un[:,:rdim], torch.diag(sn[:rdim]))
        ustot = torch.matmul(utot[:,:rdim], torch.diag(stot[:rdim]))
        Q = torch.matmul(torch.transpose(usn, 0, 1), ustot)
        uq, sq, vq = torch.linalg.svd(Q)
        theta = torch.acos(torch.abs(torch.clamp(torch.sum(sq)/torch.sum(sn[:rdim]*stot[:rdim]), min=-1.0, max=1.0)))
        return theta#/theta_w

    def optimal_thresh(self, prjct_matr, origin_zero, non_origin_zero):
        prjcts = torch.matmul(prjct_matr, origin_zero)
        non_prjcts = torch.matmul(prjct_matr, non_origin_zero)
        ratio_vls = torch.div(torch.norm(prjcts, dim = 0), torch.norm(origin_zero, dim = 0))
        non_ratio_vls = torch.div(torch.norm(non_prjcts, dim = 0), torch.norm(non_origin_zero, dim = 0))

        ref_range = 100
        true_pos_rat = []
        false_pos_rat = []
        ref_list = []
        opt_list = []

        for k in range(ref_range):
            ref_val = k/ref_range
            ref_list.append(ref_val)
            true_pos = torch.sum(ratio_vls>ref_val).item()
            false_neg = torch.sum(ratio_vls<=ref_val).item()

            true_neg = torch.sum(non_ratio_vls<=ref_val).item()
            false_pos = torch.sum(non_ratio_vls>ref_val).item()

            tpr = true_pos/(true_pos+false_neg)
            true_pos_rat.append(tpr)
            fpr = false_pos/(false_pos+true_neg)
            false_pos_rat.append(fpr)
            opt_list.append((fpr**2)+(1-tpr)**2)
            #print('tp, fn, tn, fp:',true_pos, false_neg, true_neg, false_pos)
        save_thresh = ref_list[opt_list.index(min(opt_list))]
        return(ratio_vls, non_ratio_vls, ref_list, opt_list, save_thresh, true_pos_rat, false_pos_rat)
    def get_subspace_bases(self):
        stat_tens, re_tens, nam_list = self.get_tens()
        state_tens = stat_tens[:,:8000]
        res_tens = re_tens[:8000,:]
        name_list = nam_list[:8000]
        #print(res_tens.shape)
        #taking transpose of the state tensor to make a column matrix
        # all results are looked up and affordance classes are listed
        self.afford_labellist = res_tens.unique().tolist()
        # '0' is removed since all results include it
        self.afford_labellist.remove(0)
        # defining affordance label value to name dictionary

        for i in range(len(self.afford_labellist)):
            self.afford_dict[self.afford_labellist[i]] = self.affordance_names[i]
            self.afford_dict_T[self.afford_labellist[i]] = self.affordance_names_T[i]

        non_base_point_vecs = dict()
        non_state_dict = dict()
        ratio_vals = dict()
        non_ratio_vals = dict()
        nr_of_bases_dict = dict()
        subs_angls = dict()


        red_dim = 3
        num_n = 10
        angl_list_tot = []
        if self.plot_graph:
            plt.figure()
        for i in self.afford_labellist:
            indices = torch.nonzero(torch.sum( (res_tens == i).int() , axis = 1))
            non_indices = torch.nonzero(torch.sum( (res_tens == i).int() , axis = 1)==0)
            afford_states = state_tens[:, indices.squeeze()].to(self.device)
            non_afford_states = state_tens[:, non_indices.squeeze()].to(self.device)
            self.state_dict[i] = afford_states

            mean_val = torch.mean(afford_states,1)
            non_mean_val = torch.mean(non_afford_states, 1).to(self.device)
            self.base_point_vecs[i] = mean_val.to(self.device)
            [dim1, dim2] = afford_states.shape
            base_tens = mean_val.resize(dim1, 1).expand(dim1, dim2).to(self.device)
            origin_zero_matr = (afford_states-base_tens).to(self.device)
            U, S, Vh = torch.linalg.svd(origin_zero_matr, full_matrices = True)
            ratio_tens = torch.zeros(afford_states.shape)
            non_ratio_tens = torch.zeros(non_afford_states.shape)
            all_ratio_tens = torch.zeros(torch.cat((ratio_tens, non_ratio_tens), 1).shape)

            [dim1, dim2] = non_afford_states.shape
            non_base_origin = non_mean_val.resize(dim1, 1).expand(dim1, dim2).to(self.device)
            non_origin_tozero_matr = (non_afford_states-non_base_origin).to(self.device)


            ratio_tens = torch.abs(torch.div(torch.matmul(torch.transpose(U,0,1).to(self.device), origin_zero_matr), torch.norm(origin_zero_matr, dim = 0).unsqueeze(0).expand(origin_zero_matr.shape[0],-1)))
            non_ratio_tens = torch.abs(torch.div(torch.matmul(torch.transpose(U,0,1).to(self.device), non_origin_tozero_matr), torch.norm(non_origin_tozero_matr, dim = 0).unsqueeze(0).expand(non_origin_tozero_matr.shape[0],-1)))
            state_proj_mean = torch.mean(ratio_tens, 1)
            non_state_proj_mean = torch.mean(non_ratio_tens, 1)
            indc = torch.nonzero(state_proj_mean>non_state_proj_mean).squeeze().int()
            rtls, non_ratls, ref_list, opt_list, save_thresh, true_pos_rat, false_pos_rat = self.optimal_thresh(self.project_matr( U[:,indc]).to(self.device), origin_zero_matr, non_origin_tozero_matr)
            self.threshold_dict[i] = save_thresh
            self.base_list[i] =self.project_matr( U[:, indc]).to(self.device)
            if self.plot_graph:
                plt.plot(false_pos_rat, true_pos_rat, linewidth=3.0, label = self.afford_dict_T[i])
                plt.scatter(false_pos_rat[opt_list.index(min(opt_list))], true_pos_rat[opt_list.index(min(opt_list))], color = 'r', zorder =1000, s = 60)

        if self.plot_graph:
            plt.xlabel('False Positive Ratio')
            plt.ylabel('True Positive Ratio')
            plt.grid()
            #plt.legend()
            plt.title('ROC Curves of Affordance Groups for %s'%self.model_name_T)
            plt.savefig('ROC_Curve_%s.png'%self.model_name_T.replace("/", ""))

    def load_models(self):
        self.featureExtractor = FeatureExtractorNet()
        self.objectDetector = ObjectDetectorYOLO() 

    def getWebcamPhoto(self, imagename='photo.jpg'):
        self.take_photo(imagename)

    def image_estimate(self, imagename='photo.jpg'):
        pil_img = pil_im.open(imagename)
        crops = self.objectDetector.detect_objects(imagename)
        TINT_COLOR = (0, 0, 0)  # Black
        TRANSPARENCY = .55  # Degree of transparency, 0-100%
        OPACITY = int(255 * TRANSPARENCY)
        for crop in crops:
            if 'person' in crop['label']:
                pass
            else:
                #display(Image(crop))
                crop_im = pil_im.fromarray(crop['im'][:,:,::-1])
                crop_t = self.featureExtractor.extract_features(crop_im)
                v1 = crop_t.squeeze(3).squeeze(0).to(self.device)
                m = nn.Softmax()
                r = nn.ReLU()
                prjctns = torch.tensor([(torch.norm(torch.matmul(self.base_list[x], v1-self.base_point_vecs[x]))/torch.norm(v1-self.base_point_vecs[x])).item()-0.85*self.threshold_dict[x] for x in self.afford_labellist])
                nl1 =torch.nonzero(torch.tensor([(torch.norm(torch.matmul(self.base_list[x], v1-self.base_point_vecs[x]))/torch.norm(v1-self.base_point_vecs[x])).item()>self.threshold_dict[x] for x in self.afford_labellist])).squeeze()
                sssorted, ssindices = torch.sort(prjctns)
                print('subspace order is :', [self.afford_dict[self.afford_labellist[x]] for x in ssindices.tolist()])
                r_prjctns = r(prjctns)
                r_prjctns[r_prjctns == 0] = -10e10
                output_prjctns = m(r_prjctns)

                #print(nl1)
                nla = np.argsort(-nl1)

                try:
                    nla = [self.afford_dict[self.afford_labellist[k]] for k in nla.tolist()]
                except:
                    nla = ['found none']


                ang_la = [self.curv_calc_auto(self.state_dict[k], v1).item() for k in self.afford_labellist]

                ang_ra = []
                auto_sorted, auto_indices = torch.sort(torch.tensor(ang_la))
                print('sorted angle are:', [self.afford_dict[self.afford_labellist[k]] for k in auto_indices.tolist()])
                ang_ltemp = torch.nan_to_num(torch.tensor(ang_la), nan=0)
                max_val_temp = torch.max(ang_ltemp)
                ang_la_tens = torch.nan_to_num(torch.tensor(ang_la), nan=max_val_temp)
                output_ang = m(1/ang_la_tens)
                w_weighted = torch.matmul(self.W_matr.float(), output_ang.unsqueeze(1))+ torch.matmul(self.W_matr.float(), output_prjctns.unsqueeze(1))
                #output_w = m((1/w_weighted).squeeze())
                output_w = (1/w_weighted).squeeze()
                out_sor, out_in = torch.sort(output_w)
                w_ordered = [self.afford_dict[self.afford_labellist[x]] for x in out_in.tolist()]
                print('optimal order is:', w_ordered)
                out_max = torch.max(output_w)
                output_w = (len(self.afford_labellist)-1)*output_w/out_max
                range_tens = torch.range(0, len(self.afford_labellist)-1)
                out_optim = torch.square(range_tens-torch.ones_like(range_tens)) + torch.square(output_w)
                optim_sorted, optim_ind_sorted = torch.sort(out_optim)
                w_results = torch.nonzero(output_w > 1.2*output_w[optim_ind_sorted[0]]).squeeze()
                try:
                    w_tot_results = [self.afford_dict[self.afford_labellist[x]] for x in w_results.tolist()]
                except:
                    w_tot_results = ['found none']
                print('----------------------------------------')
                print('calculated w results  are:', w_tot_results )
                print('----------------------------------------')
                box_coords = [x.item() for x in crop['box']]
                source_img = pil_im.open(imagename).convert("RGBA")
                overlay = pil_im.new('RGBA', source_img.size, TINT_COLOR+(0,))
                draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
                draw.rectangle(((box_coords[0], box_coords[1]), (box_coords[2], box_coords[3])), outline =(255, 0, 0), width = 5, fill=TINT_COLOR+(OPACITY,))
                print_string = [crop['label']] + w_tot_results
                print_string = '\n'.join(print_string)
                draw.multiline_text((box_coords[0], box_coords[1]), print_string, font =ImageFont.truetype('LiberationMono-Bold.ttf',25),fill = (255, 255, 0, 255))
                source_img = pil_im.alpha_composite(source_img, overlay)
                source_img = source_img.convert("RGB") # Remove alpha for saving in jpg format.

                plt.imshow(source_img)
                plt.show()

