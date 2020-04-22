### Simulation approach

Back in time, I used Python 2 to perform simulation and analysis. But now I would rather use Python 3.
I have prepared two definition files for singularity images.

##### Setup

First build singularity image:
```
sudo singularity build singularity/py3_activity_sequence.sif singularity/py3_activity_sequence.def
```

Then go to shell of singularity container.
```
singularity shell singularity/py3_activity_sequence.sif
```

##### Preparation

I used Sumatra to track and manage numerical simulations.

Then go to scripts folder, initialize git and smt (Remark: Project name is in this case `STAS`)
```
cd scripts
git init
smt init STAS
```

Add simulation scripts because of Sumatra.
```
git add simulation
git commit -m 'Add simulation scripts'
```


Add definition of mimetype for gdf and dat files.
```
echo 'text/plain    dat gdf' > .smt/mime.types
```


To keep tracking on current script you have to commit the changes of simulation script.


#### Simple usage of simulation script

Run simulation script (in shell)
```
smt run --main simulation/sequence_EI_networks.py params/sequence_EI_networks.json
```

Additionally, I made own functions for good protocolling method.
In `lib` folder you find a file `protocol.py` to get data if exists.
Otherwise it will perform new simulation with a set of parameters.

Run a simulation script (in ipython) with default parameters (located in `params` folder)
```
import pylab as pl
import lib.protocol as protocol

simulation = 'sequence_I_networks'
gids, ts = protocol.get_or_simulate(simulation)

fig,ax = pl.subplots(1,1)
ax.plot(ts, gids, '.')
pl.show()
```

With specific parameters

```
params = {'noise': {'mean': 800.}}
gids, ts = protocol.get_or_simulate(simulation, params)
```


#### Figures in paper

Command line to visualize results of the simulation
```
python3 ./scripts/plot/sequence_networks_schematics.py
```

List of plotting script codes in `./scripts` folder
```
Fig 1: plot/sequence_networks_schematics.py
Fig 2: plot/sequence_I_networks_measurements.py
Fig 3: plot/sequence_networks_cluster_activity.py
Fig 4: plot/sequence_networks_power_spectrum.py
Fig 5: plot/sequence_I_networks_shift_speed.py
Fig 6: plot/sequence_I_networks_perlin_size.py
Fig 7: plot/sequence_networks_mechanism.py
Fig 8: plot/sequence_EI_networks_stim_inputs.py (Attention: it takes long time.)
(Fig 9: visualized with MATLAB)

S1: plot/sequence_networks_connections.py
S3: plot/sequence_EI_networks_input_weights.py (Attention: it takes long time.)
S4: plot/sequence_EI_networks_spectrogram.py
```
