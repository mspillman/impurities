## How it works

This web-app uses a neural network model to predict if a PXRD dataset contains peaks from multiple phases (i.e. impurities) or just a single phase (pure sample). The model was trained on data designed to approximate patterns collected on laboratory-based instruments using Cu Ka1 radiation, with a step size of 0.019 degrees and a range of 4 - 44 degrees $2\theta$. Any data that does not meet these parameters clearly will need modification before it can be used by the model.

Any data collected with a different wavelength and/or step size will automatically be converted to the expected pattern for CuKa1 with the correct step size via interpolation. Any data below or above the expected range is trimmed. If the data range starts _after_ 4 degrees $2\theta$, then the first datapoint is used to fill in the blanks, which will appear as a flat line in the plots displayed. This is an augmentation that was included in the model training, and as such, the model should correctly ignore this.

Once the diffraction data is in the correct format for the model, the model is used to determine:
1. The probability that the pattern was produced by a single or multiple crystalline phases. The prediction is printed above the plot
2. An interactive plot showing the diffraction data, overlaid with the probability assigned by the model to each point on the histogram that the intensity (after accounting for the background) can be accounted for by an impurity phase.

You can upload multiple datasets at the same time, each one will get a separate plot.

The model has the following validation set performance when trained on PXRD patterns synthesised from the POW-COD database:

Best Val Pure / Impure classification:
  - Precision: 0.943
  - Recall: 0.890
  - F1: 0.892
  - Accuracy: 0.895
  - MCC 0.794

Best Val Peak-level classification:
  - Precision: 0.774
  - Recall: 0.408
  - F1: 0.511
  - Accuracy: 0.982
  - MCC 0.522