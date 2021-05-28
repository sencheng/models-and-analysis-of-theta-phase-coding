import matplotlib.pyplot as plt
from data_analysis.spikes import UniformSpikes, VariableSpikes, SpeedSpikes
from data_analysis.initializer import initialize


session = "ec014.468"
model_name = "SpeedControlled"

if 'Variable' in model_name:
    class_def = VariableSpikes
elif model_name == 'Time' or model_name == 'Position':
    class_def = UniformSpikes
else:
    class_def = SpeedSpikes


model_spikes = initialize((class_def, ), session, model_name)[class_def.__name__]
model_spikes.plot(time_interval=(800, 805))

plt.show()
