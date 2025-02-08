from torchvision.models import EfficientNet_B0_Weights,GoogLeNet_Weights,MobileNet_V2_Weights,ResNet18_Weights
from collections import OrderedDict

from networks.csta.EfficientNet import CSTA_EfficientNet
from networks.csta.GoogleNet import CSTA_GoogleNet
from networks.csta.MobileNet import CSTA_MobileNet
from networks.csta.ResNet import CSTA_ResNet

# change GoogLeNet state_dict keys to match csta.GoogleNet
def update_state_dict_keys(state_dict):
    updated_state_dict = OrderedDict()

    for old_key, value in state_dict.items():
        new_key = old_key

        # branch2 → branch2_0, branch2_1
        new_key = new_key.replace("branch2.0", "branch2_0")
        new_key = new_key.replace("branch2.1", "branch2_1")

        # branch3 → branch3_0, branch3_1
        new_key = new_key.replace("branch3.0", "branch3_0")
        new_key = new_key.replace("branch3.1", "branch3_1")

        # branch4.1 → branch4_1
        new_key = new_key.replace("branch4.1", "branch4_1")

        updated_state_dict[new_key] = value

    return updated_state_dict

# Load models depending on CNN
def set_model(model_name,
              Scale,
              Softmax_axis,
              Balance,
              Positional_encoding,
              Positional_encoding_shape,
              Positional_encoding_way,
              Dropout_on,
              Dropout_ratio,
              Classifier_on,
              CLS_on,
              CLS_mix,
              key_value_emb,
              Skip_connection,
              Layernorm,
              input_dim,
              batch_size):
    if model_name in ['EfficientNet','EfficientNet_Attention']:
        model = CSTA_EfficientNet(
            model_name=model_name,
            Scale=Scale,
            Softmax_axis=Softmax_axis,
            Balance=Balance,
            Positional_encoding=Positional_encoding,
            Positional_encoding_shape=Positional_encoding_shape,
            Positional_encoding_way=Positional_encoding_way,
            Dropout_on=Dropout_on,
            Dropout_ratio=Dropout_ratio,
            Classifier_on=Classifier_on,
            CLS_on=CLS_on,
            CLS_mix=CLS_mix,
            key_value_emb=key_value_emb,
            Skip_connection=Skip_connection,
            Layernorm=Layernorm,
            input_dim=input_dim
        )
        state_dict = EfficientNet_B0_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        model.efficientnet.load_state_dict(state_dict)
    elif model_name in ['GoogleNet','GoogleNet_Attention']:
        model = CSTA_GoogleNet(
            model_name=model_name,
            Scale=Scale,
            Softmax_axis=Softmax_axis,
            Balance=Balance,
            Positional_encoding=Positional_encoding,
            Positional_encoding_shape=Positional_encoding_shape,
            Positional_encoding_way=Positional_encoding_way,
            Dropout_on=Dropout_on,
            Dropout_ratio=Dropout_ratio,
            Classifier_on=Classifier_on,
            CLS_on=CLS_on,
            CLS_mix=CLS_mix,
            key_value_emb=key_value_emb,
            Skip_connection=Skip_connection,
            Layernorm=Layernorm,
            input_dim=input_dim,
            batch_size=batch_size
        )
        state_dict = GoogLeNet_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux')}
        state_dict = update_state_dict_keys(state_dict)
        
        new_state_dict = model.googlenet.state_dict()
        for name,param in state_dict.items():
            new_state_dict[name] = param
        model.googlenet.load_state_dict(new_state_dict)
    elif model_name in ['MobileNet','MobileNet_Attention']:
        model = CSTA_MobileNet(
            model_name=model_name,
            Scale=Scale,
            Softmax_axis=Softmax_axis,
            Balance=Balance,
            Positional_encoding=Positional_encoding,
            Positional_encoding_shape=Positional_encoding_shape,
            Positional_encoding_way=Positional_encoding_way,
            Dropout_on=Dropout_on,
            Dropout_ratio=Dropout_ratio,
            Classifier_on=Classifier_on,
            CLS_on=CLS_on,
            CLS_mix=CLS_mix,
            key_value_emb=key_value_emb,
            Skip_connection=Skip_connection,
            Layernorm=Layernorm,
            input_dim=input_dim
        )
        state_dict = MobileNet_V2_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        model.mobilenet.load_state_dict(state_dict)
    elif model_name in ['ResNet','ResNet_Attention']:
        model = CSTA_ResNet(
            model_name=model_name,
            Scale=Scale,
            Softmax_axis=Softmax_axis,
            Balance=Balance,
            Positional_encoding=Positional_encoding,
            Positional_encoding_shape=Positional_encoding_shape,
            Positional_encoding_way=Positional_encoding_way,
            Dropout_on=Dropout_on,
            Dropout_ratio=Dropout_ratio,
            Classifier_on=Classifier_on,
            CLS_on=CLS_on,
            CLS_mix=CLS_mix,
            key_value_emb=key_value_emb,
            Skip_connection=Skip_connection,
            Layernorm=Layernorm,
            input_dim=input_dim
        )
        state_dict = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        model.resnet.load_state_dict(state_dict)
    else:
        raise
    return model