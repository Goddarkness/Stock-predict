import torch
import torch.nn as nn
import torch.nn.functional as F


n_features = 6
predict_day = 3
learning_rate = 0.01


def modell(n_features, model_path=''):
    class Model_(nn.Module):
        def __init__(self, n_features):
            super(Model_, self).__init__()
            self.lstm = nn.LSTM(n_features, 128, batch_first=True)
            self.liner1 = nn.Linear(128, 64)
            self.liner2 = nn.Linear(64, predict_day)

        def forward(self, x):
            output, _ = self.lstm(x)
            output = F.relu(output)
            output = self.liner1(output)
            output = F.leaky_relu(output)
            output = self.liner2(output[:, -1, :])
            return(output)
    _model = Model_(n_features)

    if model_path:
        _model.load_state_dict(torch.load(model_path))
    else:
        for param in _model.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
            else:
                nn.init.uniform_(param, a=0.1, b=0.9)
    opt = torch.optim.SGD(_model.parameters(),
                          lr=learning_rate, weight_decay=2.0)
    criterion = nn.MSELoss()

    return _model
