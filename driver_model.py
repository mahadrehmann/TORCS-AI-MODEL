# driver_model.py

import msgParser
import carState
import carControl
import numpy as np
import torch
import joblib
import torch.nn as nn

# 1) Same feature list as in training
FEATURE_COLS = [
    'angle','curLapTime','damage','distFromStart','distRaced',
    'fuel','racePos','rpm','speedX','speedY','speedZ','trackPos','z'
] + [f'track_{i}' for i in range(19)] \
  + [f'opponent_{i}' for i in range(36)] \
  + [f'focus_{i}' for i in range(5)]

# 2) Network definition matching train_model.py
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class Driver(object):
    def __init__(self, stage, manual_mode=False):
        self.parser  = msgParser.MsgParser()
        self.state   = carState.CarState()
        self.control = carControl.CarControl()

        # Load scaler
        self.scaler = joblib.load('scaler.save')

        # Reconstruct model and load weights
        in_dim  = len(FEATURE_COLS)
        out_dim = 4   # accel, brake, steer, gear_cmd
        self.model = Net(in_dim, out_dim)
        state_dict = torch.load('torcs_model.pt', map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def init(self):
        # same init angles as other drivers
        angles = [0]*19
        for i in range(5):
            angles[i]      = -90 + i*15
            angles[18-i]   =  90 - i*15
        for i in range(5,9):
            angles[i]      = -20 + (i-5)*5
            angles[18-i]   =  20 - (i-5)*5
        return self.parser.stringify({'init': angles})

    def drive(self, msg):
        # 1) parse sensors
        self.state.setFromMsg(msg)
        s = []
        for c in FEATURE_COLS:
            if c in ['angle','curLapTime','damage','distFromStart','distRaced',
                     'fuel','racePos','rpm','speedX','speedY','speedZ','trackPos','z']:
                v = self.state.sensors.get(c, [0.0])[0]
            elif c.startswith('track_'):
                idx = int(c.split('_')[1])
                v   = self.state.sensors.get('track',    [0.0]*19)[idx]
            elif c.startswith('opponent_'):
                idx = int(c.split('_')[1])
                v   = self.state.sensors.get('opponents',[0.0]*36)[idx]
            else:  # focus_
                idx = int(c.split('_')[1])
                v   = self.state.sensors.get('focus',    [0.0]*5)[idx]
            s.append(float(v))

        # 2) scale & predict
        X   = np.array(s, dtype=np.float32).reshape(1, -1)
        Xs  = self.scaler.transform(X)
        with torch.no_grad():
            out = self.model(torch.from_numpy(Xs))
        accel, brake, steer, gear = out.numpy().flatten().tolist()

        # 3) apply clipping and set controls
        self.control.setAccel(float(np.clip(accel, 0.0, 1.0)))
        self.control.setBrake(float(np.clip(brake, 0.0, 1.0)))
        self.control.setSteer(float(np.clip(steer, -1.0, 1.0)))
        self.control.setGear(int(round(gear)))

        return self.control.toMsg()

    def onShutDown(self):
        pass

    def onRestart(self):
        pass
