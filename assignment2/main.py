# HOW TO USE

# 1. Import a model
# from model_1 import SampleArchitecture1 as model

# 2. Call the trainer with desired architecture
# from trainer import process
# process(SampleArchitecture1.get_model, "/tmp/tf/model1", batch_size=200, steps=500)


from model_2 import SampleArchitecture2 as model
from trainer import process

process(model.get_model, "/tmp/tf/modelx", batch_size=200, steps=500)
