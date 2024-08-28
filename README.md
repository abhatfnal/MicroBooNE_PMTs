# MicroBooNE_PMTs
Analysis of the dark rate and gain data for the decommissioned MicroBooNE PMTs.

In the gain.ipynb notebook, the first cell calls the BaselineCorrector class to read in the individual csv files that correspond to a single waveform recorded by the oscilloscipe. This class corrects the baseline of the waveform and extracts only the relevant part of the PMT dark noise pulse and saves all the extracted pulse in the form of npz arrays.

The second cell in the gain.ipynb jupyter notebook, we call the WaveformAnalyzer class (that reads in the npz files) to fit the waveform with different fitting functions as well as measures the peak and area under the curve from the data waveform without any fitting. It then plots the histograms corresponding to maximum height and integral of the waveforms and fits the histograms with a gaussian to determine the location of the single PE peak.
