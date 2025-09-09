## overhang_surrogates: a Python package for sampling, training and visualising surrogate models for building energy simulations

The package, which can be installed by `pip install overhang_surrogates`, contains selected methods that:
- offer Python implementation of the iterative sampling method mc-intersite-proj-th (MIPT) proposed in 
  K Crombecq, E Laermans, T Dhaene,
  *Efficient space-filling and non-collapsing sequential design strategies for simulation-based modeling*,
  Eur J Oper Res 214 (2011), 683-696,
- simplify training of XGBoost and ANN surrogate models with *k*-fold cross validation for a given pandas dataframe, and
- produce visually appealing 3D diagrams for selected columns using vedo.

It consists of the following main methods:

---

`mipt(n, dim=2, alpha=0.5, k=100)`

This function is the basic implementation of the MIPT sampling method (MIPT). Candidate points are chosen from the entire hypercube [0,1]^dim.

Parameters:
- `n` — the sample size to generate.
- `dim` — the dimension of the space from which samples are drawn.
- `alpha` — tolerance parameter for the minimal *projected* distance: any candidate point whose projected distance is less than `alpha / n` is discarded from consideration.
- `k` — factor that determines the number of candidate points (in the `i`-th iteration `k*i` candidate points are generated).

Returns: an array of `n` points from the hypercube [0,1]^dim.

---

## `mipt_full(n, dim=2, alpha=0.5, k=100, negligible=1e-6)`

This implements a version of MIPT in which new candidate points are sampled **only** from the allowed intervals obtained after removing from the hypercube \([0,1]^{\text{dim}}\) the hypercubes that cover the minimal projected distance around already selected sample points.

- `n`, `dim`, `alpha`, `k` have the same meaning as in `mipt`.
- `negligible` — a value considered negligible when comparing interval bounds (used to handle numerical edge cases).

**Returns:** an array of `n` points from the hypercube \([0,1]^{\text{dim}}\).

---

## `mipt_extend(current, n, alpha=0.5, k=100, negligible=1e-6)`

Extends an existing sample given by `current`, which is a NumPy array (e.g. `np.array([[0.5,0.5,0.5], [0.25,0.25,0.25], [0.75,0.75,0.75]])`).

- New candidate points are picked only from the allowed intervals as in `mipt_full`.
- `n`, `alpha`, `k`, `negligible` have the same meaning as above.
- `dim` is inferred from the dimensionality of the rows in the `current` NumPy array.

**Returns:** the input sample `current` augmented with `n` new points from \([0,1]^{\text{dim}}\).

---

## `add_overhang(idf, window_name, height, depth, width)`

Adds an overhang to an `eppy` IDF object `idf`.

- The overhang is placed `height` meters above the window named `window_name`.
- The overhang has depth `depth` meters and width `width` meters.
- It is centered on the window and attached to the wall that contains the window.

**Assumptions:** the window `window_name` belongs to an exterior wall and the window is rectangular with all four edges either horizontal or vertical.

---

## `create_obstacle(idf, x, y, z, dist, left, right, up)`

Creates a southern obstacle — an `EnergyPlus` object of type `Shading:Site:Detailed` — that blocks direct sunlight.

- The obstacle is created relative to the reference point `(x, y, z)` and placed `dist` meters to the south.
- The view to the south is blocked `left` degrees to the left, `right` degrees to the right, and `up` degrees upward.
- The new shading object is added to the existing `idf` object.

---

## `sample_and_simulate_overhangs(idf_filename, epw_filename, window_name, depth_range, height_range, width, sample_size, idd_filename='Energy+.idd', delete_output_files=True)`

This function:

1. Creates a sample of size `sample_size` for overhang depths and heights. Depths are chosen from `depth_range`; heights are chosen from `height_range`.
2. For each sampled (depth, height) pair it:
   - builds an `idf` object from `idf_filename` and `epw_filename`,
   - adds the corresponding overhang to that `idf`,
   - simulates the `idf` using the `eppy` library,
   - collects values of the meters defined in the `idf` via `Output:Meter` commands (e.g. `Output:Meter,DistrictHeating:Facility,RunPeriod;`).

3. Combines all collected meter values into a single `pandas` `DataFrame`, indexed by the sampled depths and heights, with columns corresponding to meters present in `idf_filename`.

**Other parameter notes**
- `window_name` — the name of the window above which overhangs are placed.
- `depth_range` — array of possible overhang depths (meters) from which the sample is drawn. If you have `min_depth`, `max_depth`, and `num_of_values`, you can generate it as:
  ```python
  depth_range = np.linspace(min_depth, max_depth, num_of_values).round(decimals=2)
  ```
  The rounding (`round(decimals=2)`) is suggested for nicer display (fewer unnecessary decimals).
- `height_range` — analogous array for possible overhang heights; same generation suggestion applies.
- `width` — the overhang width (meters).
- `idd_filename` — path to the EnergyPlus IDD file. Default `'Energy+.idd'` implies a copy of this file should exist in the current directory.
- `delete_output_files` — whether to delete files produced by EnergyPlus after the simulations. Set to `False` only if you need to inspect those files afterwards.

**Returns:** a `pandas.DataFrame` with all collected meter predictions.

---

## `train_model_ensembles(df, input_cols, meter_cols, num_folds=5, learning_rate=0.1, early_stopping_rounds=10)`

Given a `pandas.DataFrame` `df`, this function trains models using the columns in `input_cols` as inputs. For each column listed in `meter_cols` it trains a separate **XGBoost ensemble**.

- Training uses mean squared error (MSE) as the loss function.
- Cross-validation with `num_folds` splits is used; therefore for each output column the ensemble is represented as a list of `num_folds` XGBoost models.
- `learning_rate` controls the XGBoost learning rate — smaller values allow finer fit but typically require more trees.
- Training stops early after `early_stopping_rounds` iterations if there is no improvement in predictive accuracy.

**Returns:** a dictionary of XGBoost ensembles; each key is a meter name from `meter_cols` and the value is the corresponding ensemble (list of models).

---

## `predict_meters(ensembles, input_ranges)`

- `ensembles` — a dictionary where each key is a meter name and each value is the XGBoost ensemble that predicts that meter. The ensemble prediction is the mean of predictions of all models in the ensemble.
- `input_ranges` — a dictionary where each key is an input variable name and each value is an array of possible values for that input.

**Behavior**
- The function forms the Cartesian product of all input value arrays.
- For each meter it predicts values for all combinations of input values.
- Returns a `pandas.DataFrame` containing all predictions.

**Example**
```python
input_ranges = {
  'depth':  np.linspace(0.1, 1.0, 10).round(decimals=1),
  'height': np.linspace(0.1, 0.5, 5).round(decimals=1)
}
```
This would create a Cartesian product of the two arrays (50 `(depth, height)` pairs), predict for each pair, and return a DataFrame with the predictions.

---

## `make_3d_diagram(df, xcol, ycol, zcol, minxvalue=None, maxxvalue=None, numxticks=10, minyvalue=None, maxyvalue=None, numyticks=10, minzvalue=None, maxzvalue=None, numzticks=10, filename=None, light=None, camera=(-15, 25), aspect_ratio=(1, 1, 1), palette=colorcet.CET_L20, distance_multiplier=4, num_isolines=25, padding=0.001, interactive=False, diagram_size='auto', **kwargs)`

Creates a 3D diagram for the data contained in columns `xcol`, `ycol` and `zcol` of the `pandas.DataFrame` `df`.

**Assumption:** the DataFrame contains exactly one row for every pair in the Cartesian product of the unique values that appear in `xcol` and `ycol`.

- The final diagram is saved to `filename`. If `interactive=True`, the diagram is displayed in interactive mode using `vedo`.

**Parameter descriptions**
- `minxvalue` — smallest x-axis tick. If `None`, set to `min(df[xcol])`.
- `maxxvalue` — largest x-axis tick. If `None`, set to `max(df[xcol])`.
- `numxticks` — number of ticks on the x-axis (including min and max).
- `minyvalue`, `maxyvalue`, `numyticks` — analogous settings for the y-axis.
- `minzvalue`, `maxzvalue`, `numzticks` — analogous settings for the z-axis.
- `filename` — output filename for the diagram. If `None`, the file will be written in the current directory as:
  ```
  diagram_<xcol>_<ycol>_<zcol>.png
  ```
  (If a file with that name already exists it will be overwritten.)
- `light` — a `vedo` light object (`Light`). If `None`, a light is constructed internally to illuminate the plot from above.
- `camera` — either a `vedo` camera parameter dictionary or a pair `(azimuth, height)` in degrees:
  - `height` is the angle between the xy-plane and the camera position,
  - `azimuth` is the angle in the xy-plane between the camera position and south (negative y-axis).
- `aspect_ratio` — ratio between x-width, y-depth and z-height of the plot. Internally normalized so the smallest value equals 1.
- `palette` — a colormap for the diagram. See Colorcet continuous maps (recommended: `colorcet.CET_L3`, `CET_L7`, `CET_L17`, `CET_L20`).
- `distance_multiplier` — if `camera` is given as `(azimuth, height)`, this sets the distance from the diagram centre as multiples of half the diagonal of the bounding parallelepiped.
- `num_isolines` — number of isolines on the diagram.
- `padding` — value added on both sides of x, y and z ranges to ensure that ticks at extreme values are shown (`padding` is added to the normalized `aspect_ratio`).
- `interactive` — if `True`, use vedo’s interactive display; otherwise produce a static image.
- `diagram_size` — size used for the vedo `Plotter`.
- `kwargs` — dictionary of extra parameters forwarded directly to the `Plotter` object.

**Returns:** nothing (the function saves or displays the diagram as specified).


