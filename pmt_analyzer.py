import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import re
from tqdm import tqdm

class WaveformAnalyzer:
    def __init__(self, base_directory, pdf_histograms_filename, results_csv_filename, display_waveforms=False, num_waveforms_to_process=None):
        self.base_directory = Path(base_directory)
        self.pdf_histograms_filename = pdf_histograms_filename
        self.results_csv_filename = results_csv_filename
        self.display_waveforms = display_waveforms
        self.num_waveforms_to_process = num_waveforms_to_process

    def gaussian_pmt_signal(self, t, A, t0, sigma, A_offset):
        return A * np.exp(-((t - t0)**2 / (2 * sigma**2))) + A_offset

    def lorentzian(self, t, A, t0, gamma, A_offset):
        return A / (1 + ((t - t0) / gamma)**2) + A_offset

    def log_normal(self, t, A, t0, tau, sigma):
        with np.errstate(divide='ignore', invalid='ignore'):
            valid_mask = (t - t0 > 0)
            y = np.zeros_like(t)
            y[valid_mask] = A * np.exp(-0.5 * ((np.log((t[valid_mask] - t0) / tau) / sigma)**2))
            return y

    def baseline_correction(self, data):
        baseline = np.mean(data[:50])
        return data - baseline

    def calc_reduced_chi_squared(self, y_obs, y_exp, dof):
        y_exp = np.where(y_exp == 0, np.finfo(float).eps, y_exp)
        residuals = y_obs - y_exp
        chi_squared = np.sum((residuals**2) / np.abs(y_exp))
        reduced_chi_squared = chi_squared / dof
        return reduced_chi_squared

    def robust_fit(self, fit_func, time_focused, amplitude_focused, initial_guess, bounds):
        def residuals(params):
            return amplitude_focused - fit_func(time_focused, *params)
        result = least_squares(residuals, initial_guess, bounds=bounds, loss='soft_l1')
        return result.x

    def process_waveforms(self, waveforms, pmt, voltage, fit_window=(-30, 30)):
        results = []
        waveforms_to_process = dict(list(waveforms.items())[:self.num_waveforms_to_process]) if self.num_waveforms_to_process else waveforms
        for waveform_key, waveform_data in tqdm(waveforms_to_process.items(), desc=f"Processing {pmt} {voltage}"):
            try:
                time_focused, amplitude_focused = waveform_data

                initial_guess_gaussian = [-0.5, 0, 5, 0]
                initial_guess_lorentzian = [-0.5, 0, 5, 0]
                initial_guess_log_normal = [-0.5, 0, 5, 1]

                bounds_gaussian = ([-1.0, -20, 1, -0.1], [0, 20, 20, 0.1])
                bounds_lorentzian = ([-1.0, -20, 1, -0.1], [0, 20, 20, 0.1])
                bounds_log_normal = ([-1.0, -20, 1, 0.1], [0, 20, 20, 10])

                fits = {}

                if len(time_focused) == 0 or len(amplitude_focused) == 0:
                    raise ValueError("Empty waveform data")

                try:
                    popt_gaussian = self.robust_fit(self.gaussian_pmt_signal, time_focused, amplitude_focused, initial_guess_gaussian, bounds_gaussian)
                    fits["Gaussian"] = (self.gaussian_pmt_signal, popt_gaussian)
                except Exception as e:
                    print(f"Gaussian fit did not converge: {e}")

                try:
                    popt_lorentzian = self.robust_fit(self.lorentzian, time_focused, amplitude_focused, initial_guess_lorentzian, bounds_lorentzian)
                    fits["Lorentzian"] = (self.lorentzian, popt_lorentzian)
                except Exception as e:
                    print(f"Lorentzian fit did not converge: {e}")

                try:
                    popt_log_normal = self.robust_fit(self.log_normal, time_focused, amplitude_focused, initial_guess_log_normal, bounds_log_normal)
                    fits["Log Normal"] = (self.log_normal, popt_log_normal)
                except Exception as e:
                    print(f"Log Normal fit did not converge: {e}")

                best_fit_name = None
                best_r_squared = -np.inf

                # Calculate parameters for each fit
                for fit_name, (fit_func, popt) in fits.items():
                    y_fit = fit_func(time_focused, *popt)
                    residuals = amplitude_focused - y_fit
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((amplitude_focused - np.mean(amplitude_focused))**2)
                    r_squared = 1 - (ss_res / ss_tot)

                    reduced_chi_squared = self.calc_reduced_chi_squared(amplitude_focused, y_fit, len(time_focused) - len(popt))

                    if y_fit.size > 0:
                        min_height = np.min(y_fit)
                    else:
                        min_height = np.nan

                    peak_position = popt[1]
                    mask_integral = (time_focused >= peak_position - 20) & (time_focused <= peak_position + 20)
                    integral_data = np.trapz(amplitude_focused[mask_integral], time_focused[mask_integral])

                    result = {
                        'waveform': waveform_key,
                        'fit_name': fit_name,
                        'min_height': min_height,
                        'integral_fit': integral_data,
                        'r_squared': r_squared,
                        'reduced_chi_squared': reduced_chi_squared
                    }

                    results.append(result)

                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_fit_name = fit_name

                # Add best fit information
                if best_fit_name is not None:
                    best_fit = {
                        'waveform': waveform_key,
                        'fit_name': 'Best Fit',
                        'min_height': results[-1]['min_height'],
                        'integral_fit': results[-1]['integral_fit'],
                        'r_squared': results[-1]['r_squared'],
                        'reduced_chi_squared': results[-1]['reduced_chi_squared'],
                        'best_fit_name': best_fit_name
                    }
                    results.append(best_fit)

                # Calculate min height and integral from the data itself
                if amplitude_focused.size > 0:
                    data_min_height = np.min(amplitude_focused)
                    data_integral = np.trapz(amplitude_focused, time_focused)
                    data_result = {
                        'waveform': waveform_key,
                        'fit_name': 'Data',
                        'min_height': data_min_height,
                        'integral_fit': data_integral,
                        'r_squared': np.nan,
                        'reduced_chi_squared': np.nan
                    }
                    results.append(data_result)

                if self.display_waveforms:
                    self.plot_waveform(time_focused, amplitude_focused, fits, waveform_key, pmt, voltage)
            except Exception as e:
                print(f"Error processing {waveform_key}: {e}")

        df_results = pd.DataFrame(results)
        return df_results

    def plot_histograms(self, results_df, pmt, voltage, pdf):
        if 'fit_name' not in results_df.columns:
            return

        min_heights_gaussian = results_df[results_df['fit_name'] == 'Gaussian']['min_height']
        integrals_gaussian = results_df[results_df['fit_name'] == 'Gaussian']['integral_fit']
        min_heights_lorentzian = results_df[results_df['fit_name'] == 'Lorentzian']['min_height']
        integrals_lorentzian = results_df[results_df['fit_name'] == 'Lorentzian']['integral_fit']
        min_heights_log_normal = results_df[results_df['fit_name'] == 'Log Normal']['min_height']
        integrals_log_normal = results_df[results_df['fit_name'] == 'Log Normal']['integral_fit']
        min_heights_best_fit = results_df[results_df['fit_name'] == 'Best Fit']['min_height']
        integrals_best_fit = results_df[results_df['fit_name'] == 'Best Fit']['integral_fit']
        min_heights_data = results_df[results_df['fit_name'] == 'Data']['min_height']
        integrals_data = results_df[results_df['fit_name'] == 'Data']['integral_fit']

        BINS = 100

        plt.figure(figsize=(14, 22))

        plt.subplot(5, 2, 1)
        plt.hist(min_heights_gaussian.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Gaussian')
        plt.xlabel('Min Height (V)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Min Heights (Gaussian Fit)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.subplot(5, 2, 2)
        plt.hist(integrals_gaussian.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Gaussian')
        plt.xlabel('Integral of Fit (V*ns)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Integrals (Gaussian Fit)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.subplot(5, 2, 3)
        plt.hist(min_heights_lorentzian.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Lorentzian')
        plt.xlabel('Min Height (V)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Min Heights (Lorentzian Fit)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.subplot(5, 2, 4)
        plt.hist(integrals_lorentzian.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Lorentzian')
        plt.xlabel('Integral of Fit (V*ns)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Integrals (Lorentzian Fit)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.subplot(5, 2, 5)
        plt.hist(min_heights_log_normal.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Log Normal')
        plt.xlabel('Min Height (V)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Min Heights (Log Normal Fit)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.subplot(5, 2, 6)
        plt.hist(integrals_log_normal.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Log Normal')
        plt.xlabel('Integral of Fit (V*ns)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Integrals (Log Normal Fit)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.subplot(5, 2, 7)
        plt.hist(min_heights_best_fit.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Best Fit')
        plt.xlabel('Min Height (V)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Min Heights (Best Fit)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.subplot(5, 2, 8)
        plt.hist(integrals_best_fit.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Best Fit')
        plt.xlabel('Integral of Fit (V*ns)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Integrals (Best Fit)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.subplot(5, 2, 9)
        plt.hist(min_heights_data.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Data')
        plt.xlabel('Min Height (V)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Min Heights (Data)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.subplot(5, 2, 10)
        plt.hist(integrals_data.dropna(), bins=BINS, alpha=1, histtype='stepfilled', label='Data')
        plt.xlabel('Integral of Fit (V*ns)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Integrals (Data)\n{pmt} at {voltage}')
        plt.grid(True)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

    def plot_waveform(self, time, amplitude, fits, waveform_key, pmt, voltage):
        plt.figure(figsize=(10, 6))
        plt.plot(time, amplitude, label='Data',color='k')
        colors = {'Gaussian': 'red', 'Lorentzian': 'orange', 'Log Normal': 'green'}
        for fit_name, (fit_func, popt) in fits.items():
            plt.plot(time, fit_func(time, *popt), label=f'{fit_name} fit', color=colors[fit_name],linestyle='--',lw=3)
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude (V)')
        plt.title(f'Waveform {waveform_key} - {pmt} at {voltage}')
        plt.legend()
        plt.show()

    def run_analysis(self):
        try:
            with PdfPages(self.pdf_histograms_filename) as pdf_histograms:
                pmt_results = []
                for npz_file in sorted(self.base_directory.glob('*_waveforms.npz'), key=lambda x: int(re.search(r'\d+', x.stem).group())):
                    pmt = npz_file.stem.split('_')[0]
                    print(f"Processing {pmt}")  # Information about the PMT being analyzed
                    data = np.load(npz_file, allow_pickle=True)
                    voltage_data = {key: value for key, value in data.items()}
                    for voltage in voltage_data:
                        print(f"Processing {pmt} {voltage}")  # Information about the voltage being analyzed
                        waveforms = voltage_data[voltage].item()  # Using .item() to get the dictionary from the array
                        results_df = self.process_waveforms(waveforms, pmt, voltage)
                        if not results_df.empty:  # Save results only if they exist
                            pmt_results.append(results_df)
                            self.plot_histograms(results_df, pmt, voltage, pdf_histograms)

                results_df_all = pd.concat(pmt_results, ignore_index=True)
                results_df_all.to_csv(self.results_csv_filename, index=False)
                np.savez_compressed(f'{self.base_directory}/waveform_results.npz', pmt_results=results_df_all.to_dict('records'))
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
