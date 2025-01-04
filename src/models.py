import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict, defaultdict
from functions.constants import MODEL_OUTPUT_ACTIVATIONS


class TorquePredictorFF(nn.Module):
    def __init__(self, 
                 input_features, 
                 hidden_layers_arch=[128, 64, 32], 
                 output_features=1,
                 output_activation=nn.Tanh) -> None:
        super(TorquePredictorFF, self).__init__()

        self.ARCH = [*hidden_layers_arch]

        # Initialize hidden layers
        layers = []

        # Counts keeps a track of how many of a certain layer we have
        counts = defaultdict(int)
        
        def add(name: str, layer: nn.Module) -> None:
            # Convertng hidden layers into an Ordered Dict down below
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        layer_in = input_features
        
        for x in self.ARCH:
            # linear-relu
            add("Linear", nn.Linear(layer_in, x))
            add("Dropout", nn.Dropout(0.2))
            add("ReLU", nn.ReLU(True))
            layer_in = x

        out_activation_name = self._get_output_activation_name(output_activation)

        # Add output torque predictor with tanh layer since 
        # torque is normalized (in the commaSteeringControl dataset)
        out_layers = [
          ('LinearOut', nn.Linear(hidden_layers_arch[-1], output_features)),
          (out_activation_name, output_activation())
        ]
        
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.output_predictor = nn.Sequential(OrderedDict(out_layers))

        self.double()

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.output_predictor(x)
        
        return x

    def _get_output_activation_name(self, output_activation):
        try:
            # if output_activation is an instance of nn.Module, call type()
            if isinstance(output_activation, nn.Module):
                return MODEL_OUTPUT_ACTIVATIONS[type(output_activation)]
            
            return MODEL_OUTPUT_ACTIVATIONS[output_activation]
        except KeyError as e:
            # TODO: Add logging support
            print(f"ERROR: Output activation {output_activation} not supported")
            print("INFO: Please choose from the following activations")
          
            for k in MODEL_OUTPUT_ACTIVATIONS.keys():
                print(k)

            raise e


class TorquePredictorLSTM(nn.Module):
    def __init__(self, 
                 input_features, 
                 hidden_size=128, 
                 num_layers=2, 
                 output_features=1, 
                 return_sequences=False,
                 dropout=0.2) -> None:
        super(TorquePredictorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.lstm = nn.LSTM(input_features, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_features)

        self.double()

    def forward(self, x: torch.Tensor):
        lstm_out, (h_n, _) = self.lstm(x)


        # .view(-1, self.hidden_size) flattens the tensor
        # if self.return_sequences:
        #     # Sequence-to-sequence behavior: pass all time steps through the fully connected layer
        #     x = self.fc(lstm_out)  # Shape: [batch_size, seq_length, output_features]
        #     x = nn.Dropout(0.2)(h_n[-1].view(-1, self.hidden_size))
        #     x = self.fc(h_n[-1])
        # else:
        #     x = nn.Dropout(0.2)(h_n[-1].view(-1, self.hidden_size))
        #     x = self.fc(h_n[-1])

        x = self.fc(lstm_out)
        # x = nn.Tanh()(x)

        return x
        


MODEL_TYPES = {
    TorquePredictorFF: 'ff',
    TorquePredictorLSTM: 'lstm'
}