import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

class BaselineCorrector:
    def __init__(self, base_directory, fit_window=(-30, 30)):
        self.base_directory = Path(base_directory)
        self.fit_window = fit_window
    
    def baseline_correction(self, data):
        baseline = np.mean(data[:50])
        return data - baseline
    
    def process_file_adjusted(self, file_path):
        try:
            df = pd.read_csv(file_path, skiprows=5, names=['time', 'amplitude'])
            df = df.dropna()
            time = df['time'].astype(float).values * 1e9
            amplitude = df['amplitude'].astype(float).values

            # Correct baseline
            amplitude_corrected = self.baseline_correction(amplitude)
            
            # Focus on region around the pulse using the specified fit window
            mask = (time >= self.fit_window[0]) & (time <= self.fit_window[1])
            time_focused = time[mask]
            amplitude_focused = amplitude_corrected[mask]
            
            return time_focused, amplitude_focused
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None

    def save_baseline_corrected_waveforms(self):
        for pmt_directory in sorted(self.base_directory.iterdir(), key=lambda x: int(re.search(r'\d+', x.name).group()) if re.search(r'\d+', x.name) else float('inf')):
            if not pmt_directory.is_dir():
                continue
            pmt = pmt_directory.name
            voltage_data = {}

            for voltage_directory in sorted([d for d in pmt_directory.iterdir() if d.is_dir()], key=lambda x: int(re.search(r'\d+', x.name).group())):
                voltage = voltage_directory.name
                waveforms = {}

                csv_files = sorted(list(voltage_directory.rglob('*.csv')), key=lambda f: int(re.search(r'(\d+)$', f.stem).group()))

                for file_counter, file_path in enumerate(tqdm(csv_files, desc=f'Processing {pmt} {voltage}')):
                    time_focused, amplitude_focused = self.process_file_adjusted(file_path)
                    if time_focused is not None and amplitude_focused is not None:
                        waveforms[f'waveform_{file_counter}'] = np.vstack((time_focused, amplitude_focused))

                voltage_data[voltage] = waveforms

            np.savez_compressed(f'{pmt}_waveforms.npz', **voltage_data)


