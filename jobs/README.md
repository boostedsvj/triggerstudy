## Producing the trigger study columns for background

Processing the entire background requires using jobs.

First install jdlfactory:

```bash
pip install jdlfactory
```

Then run (don't forget to renew your grid proxy):

```bash
python submit.py
cd bkgcols
condor_submit
```

When the jobs are finished (should take a few hours), run the concatenation script to combine all the separate column files into single ones should take about 15 min):

```bash
python concat.py root://cmseos.fnal.gov//store/user/lpcdarkqcd/triggerstudy/bkg_Dec08/TRIGCOL/Summer20UL18/*
```
