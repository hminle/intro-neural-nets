# HOW TO USE

# 1. Import a model
# from model_1 import SampleArchitecture1 as model

# 2. Set parameters (model_dir, batch_size, steps)
# 3. Execute program with `python3 main.py`

# import os
# import sys
#
# pwd = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, pwd + "/deps")

# Import desired model
from model_1 import SampleArchitecture1 as used_model
from trainer import process
import time

start_time = time.time()

model = used_model.get_model

# Change these parameters
model_dir = "/tmp/tf/model"
use_gpu = False
batch_size = 200
steps = 500

process(model, model_dir, batch_size=batch_size, steps=steps, use_gpu=use_gpu)

end_time = time.time()
print(f"--- {end_time - start_time} seconds ---")
