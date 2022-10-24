# Trigger 

## Setup

```
python3 -m venv env
source env/bin/activate  # Needed every time
pip install numpy uproot seutils matplotlib requests
```


## Checking the madpt cut

Download the samples with various boost cuts:

```
xrdcp -r root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/orthogonalitystudy/HADD .
mv HADD boost_samples
```

Turn the root files into dumb columns stored in a .npz file:

```
python test.py make_npzs_new -r boost_samples/madpt200_mz250_mdark10_rinv0.3.root -r boost_samples/madpt0_mz250_mdark10_rinv0.3.root -r boost_samples/madpt400_mz250_mdark10_rinv0.3.root -o boost_samples/
```

Plot the AK reco jet pt distribution:

```
python test.py plot_jetptdist boost_samples/madpt*.npz
```

![pt distribution madpt](example_plots/madpt_ptjet_distribution.png)

`madpt0` and `madpt200` are quite alike, except that `madpt200` has way more statistics; this is expected, as there is a 170 GeV jet cut-off in TreeMaker. `madpt400` starts way higher into the spectrum, as expected. Zoom in on the tails:

```
python test.py plot_jetptdist boost_samples/madpt*.npz --highptzoomin
```

![pt distribution madpt](example_plots/madpt_ptjet_distribution_highpt.png)

If the madpt cut is orthogonal to the AK8 jet pt > 500 cut, we would expect identical distributions here. Clearly the `madpt0` sample has too few statistics. It looks like `madpt400` is slightly different from `madpt200` in the very low end of the distribution, so perhaps it is slightly too aggressive. A madpt cut of 300 seems like a reasonable tradeoff between not generating too many useless events (AK8 jet pt < 500) and not modifying the distributions that survive the trigger threshold cut (AK8 jet pt >= 500).