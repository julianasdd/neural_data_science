import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define filter parameters
fs = 1000  # Sampling frequency in Hz
fc = 100   # Cutoff frequency in Hz
order = 4  # Filter order

# Create frequency axis (log scale for better visualization)
frequencies = np.logspace(0, 3, num=500)  # 500 frequency points from 1 Hz to 1000 Hz

# Butterworth filter (sharp roll-off)
b, a = signal.butter(order, fc, fs=fs, btype='low')
w_butter, h_butter = signal.freqs(b, a, worN=frequencies * 2 * np.pi)

# Bessel filter (gradual roll-off)
b_bessel, a_bessel = signal.bessel(order, fc, fs=fs, btype='low')
w_bessel, h_bessel = signal.freqs(b_bessel, a_bessel, worN=frequencies * 2 * np.pi)

# Plot the frequency response of both filters
plt.figure(figsize=(10, 6))

# Butterworth Filter (Sharp Roll-off)
plt.semilogx(w_butter / (2 * np.pi), 20 * np.log10(np.abs(h_butter)), label="Butterworth Filter", linewidth=2)

# Bessel Filter (Gradual Roll-off)
plt.semilogx(w_bessel / (2 * np.pi), 20 * np.log10(np.abs(h_bessel)), label="Bessel Filter", linestyle='--', linewidth=2)

# Highlight the cutoff frequency
plt.axvline(fc, color='red', linestyle=':', label=f"Cutoff Frequency {fc} Hz")

# Labels and Title
plt.title("Frequency Response: Butterworth vs Bessel Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which="both", ls="--")
plt.legend()

# Show plot
plt.tight_layout()
plt.show()