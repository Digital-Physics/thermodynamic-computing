This is a repo to explore "Thermodynamic Computing" with the thrml library from Extropic.

Everything has been vibe coded.  

I tried to convert some evolutionary code that is effective for solving a Conway's Game of Life pattern matching game into a Thermodynamic approach for finding an action sequence. It still needs work.

```
uv run thermo_ca.py --pattern-file target_pattern.npy  --iterations 50 --steps 10 --live-plot  --beta-initial 0.1 --beta-final 10.0
```

```
uv run demo_thermo.py  --sequence-file best_sequence_thermo.npy  --pattern-file target_pattern.npy  --grid-size 12  --rules conway
```