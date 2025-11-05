This is a repo to explore "Thermodynamic Computing" with the thrml library from Extropic.

Everything has been vibe coded thus far. The main.py and different_social_influence_betas.py files are to get some bearings and learn the fundamental concepts.

I tried to convert some evolutionary code that I developed that is effective for finding an action sequence solution for a [Conway's Game of Life pattern matching game](https://evolutionary-ca-webgpu.onrender.com/) into a Thermodynamic approach for finding an action sequence solution. It still needs work.

```
uv run thermo_ca.py --pattern-file target_pattern.npy  --iterations 50 --steps 10 --live-plot  --beta-initial 0.1 --beta-final 10.0
```

```
uv run demo_thermo.py  --sequence-file best_sequence_thermo.npy  --pattern-file target_pattern.npy  --grid-size 12  --rules conway
```