# modifying EfficientNEt by adding Dilated Convolutions
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

class DN(nn.Module):
    def __init__(self, input_channels, output_channels, dilations, filters):
        super(DN, self).__init__()

        assert len(dilations) == len(filters)


        #list of parallel dilated convolution layers
        self.parallel_convs = nn.ModuleList([
            nn.Conv2d(in_channels=input_channels,
                             out_channels=temp,
                              kernel_size=1,
                               dilation = dilations[0],
                                padding=dilations[0]),

            nn.Conv2d(in_channels=temp,  #temp
                       out_channels=output_channels,
                        kernel_size=f,
                          dilation=d,
                            padding=d)
            for d, f in zip(dilations[1:], filters[1:])
        ])

        #combining parallel outputs through sum
        self.combine = nn.Conv2d(output_channels * len(dilations),
                                 output_channels,
                                 kernel_size=1)

    def forward(self, x):
        # applying all parallel convolutions and concatenating their output
        out = torch.cat([conv(x) for conv in self.parallel_convs], dim=1)
        out = self.combine(out)
        return out
    

class ModifiedEfficientNetB7(EfficientNet):
    def __init__(self):
        super(ModifiedEfficientNetB7, self).__init__(blocks_args=None, global_params=None)
        self.dn1 = DN(input_channels=round_filters(320,self._global_params),  #E5 output
                      output_channels=round_filters(512, self._global_params), #E6 input
                      dilations=[1,2,3,4], filters=[3,5,3,5])  
        
        self.dn2 = DN(input_channels=round_filters(512,self._global_params),   #E6 output
                      output_channels=round_filters(640, self._global_params), #E7 input
                      dilations=[1,2,3,4], filters=[3,5,3,5])
        
    def extract_features(self, inputs):
        # retuen output of the final conv layer
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        for idx,block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            
            # Forward pass through the block
            x = block(x, drop_connect_rate=drop_connect_rate)

            #inserting Dn1 b/w E5 and E6
            if idx == 4:
                x = self.dn1(x)

            #inserting Dn2 b/w E6 and E7
            if idx == 5:
                x = self.dn2(x)

        x = self._swish(self._bn1(self._conv_head(x)))
        return x


# Utility function to load pretrained EfficientNet-B7
def load_modified_efficientnet_b7():
    # Load pretrained EfficientNet-B7 model
    model = EfficientNet.from_pretrained('efficientnet-b7')
    
    # Initialize modified EfficientNet-B7 and load its weights into the modified architecture
    modified_model = ModifiedEfficientNetB7()
    modified_model.load_state_dict(model.state_dict(), strict=False)  # Allow missing keys for added layers

    return modified_model


# Example usage
if __name__ == "__main__":
    # Load the modified EfficientNet-B7 model
    model = load_modified_efficientnet_b7()

    # Create dummy input tensor
    inputs = torch.randn(1, 3, 224, 224)

    # Forward pass through the model
    outputs = model.extract_features(inputs)
    print(outputs.shape)
