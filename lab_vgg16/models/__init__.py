from .vgg import *
from .resnet import *
from .saperate_vgg.vgg_cnn import VGG_CNN
from .saperate_vgg.vgg_fc import VGG_FC
from .resnet_cnn import ResCNN18,ResCNN34,ResCNN50,ResCNN101,ResCNN152
from .resnet_feat2 import Feat2_ResNet18,Feat2_ResNet34,Feat2_ResNet50,Feat2_ResNet101,Feat2_ResNet152
from .resnet_lsoftmax import Lsoftmax_ResNet34, Lsoftmax_ResNet18
from .resnet_arcface import ArcMargin_ResNet34, ArcMargin_ResNet18
from .resnet_addmargin import AddMargin_ResNet34, AddMargin_ResNet18
from .resnet_sphereface import SphereProduct_ResNet34, SphereProduct_ResNet18
from .weight_resnet import WResNet18, WResNet34
from .vgg_cnn import VGG16_CNN
from .vgg_feat2 import Feat2_VGG16
from .vgg_lsoftmax import Lsoftmax_VGG16
from .vgg_arcface import ArcMargin_VGG16
from .vgg_sphereface import SphereProduct_VGG16
from .vgg_addmargin import AddMargin_VGG16
from .weight_vgg import WVGG16
from .weight_vgg_feat import WVGG16_feat
from .resnet_PW import PWResNet34
from .vgg_PW import PWVGG16