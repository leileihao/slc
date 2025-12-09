import os
import sys
import re
import numpy as np
import scipy.io as so
import scipy.signal as ssig

def get_sr(ppath, name):
    """
    read and return sampling rate (SR) from the info.txt file $ppath/$name/info.txt
    """
    fid = open(os.path.join(ppath, name, 'info.txt'), newline=None)
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + 'SR' + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))
    return float(values[0])

def my_bpfilter(x, w0, w1, N=4, bf=True):
    """
    create N-th order bandpass Butterworth filter with corner frequencies
    w0*sampling_rate/2 and w1*sampling_rate/2
    """

    b, a = ssig.butter(N, [w0, w1], 'bandpass')
    if bf:
        y = ssig.filtfilt(b, a, x)
    else:
        y = ssig.lfilter(b, a, x)

    return y

def smooth_data(x, sig):
    """
    y = smooth_data(x, sig)
    Smooth data vector x with a Gaussian kernel
    using standard deviation sig.
    Reflective padding is used to avoid edge distortion.
    """
    sig = float(sig)
    if sig == 0.0:
        return x

    # Gaussian kernel
    gauss = lambda x, sig: np.exp(-(x * x) / (2. * sig * sig))

    # Kernel radius: large enough to cover essentially all Gaussian weight
    L = int(np.ceil(4 * sig))  # ~99.99% of Gaussian mass
    xs = np.arange(-L, L + 1.)
    F = gauss(xs, sig)
    F /= F.sum()  # normalize

    # Reflect padding to avoid edge distortion
    pad = len(F) // 2
    x_padded = np.pad(x, pad, mode='reflect')

    # Convolve and crop back
    y = ssig.fftconvolve(x_padded, F, mode='same')
    y = y[pad:-pad]  # remove padding

    return y

def find_hr(ppath, recording, bin_s=2.5, recalculate = False, emg_select = 'EMG2', filter_percent = 0.1, name='none'):
    """
    Calculates the heartrate for a recording (or loads the previously saved heartrate)
    saves as hr.mat
    ppath -> file location
    recording -> recording name
    bin_s -> length of bin for calculating heartrate
    recalculate -> ignore previously saved hr data and rerun peak detection
    emg_select -> name of channel that has the R-wave peaks of the heartbeat
    filter_percent -> maximum allowed percentage of bad beats in a bin to still consider it 'good'
    name -> name to save hr data as (default 'none' saves as hr.mat)

    Returns:
        heartrate -> calculated heartrate (bpm) for each bin
        goodbin -> which bins were successfully calculated
        idxs -> indices of the R-waves of heartbeats (corresponding to the sample index of the EMG)
        thresholds -> optimal thresholds used for each bin
    """
    if name == 'none':
        hr_file = os.path.join(ppath, recording, 'hr.mat')
    else: hr_file = os.path.join(ppath, recording, name + '.mat')

    #Check if hr data has already been calculated
    #if not, calculate and save
    if not os.path.exists(hr_file) or recalculate:
        attempts = 0
        prev_quality = np.inf
        print('calculating hr for ' + recording)

        while attempts < 3:
            #constants
            sr = get_sr(ppath, recording)
            nbin = int(np.round(sr) * bin_s)
            dt = (1.0 / sr) * nbin #unused
            min_bpm = 300
            max_bpm = 1000
            max_bps = max_bpm / 60  #1000 bpm is upper limit for mouse (at least under regular observation in a box)
            min_bps = min_bpm / 60  #300 bpm is lower limit for mouse
            min_dis = (1 / max_bps) * sr
            min_dis_singlehb = .01 * sr #an individual heartbeat is ~10ms in duration
            min_beats = min_bps * bin_s #minimum number of beats in a bin corresponding to 300 bpm
            max_beats = max_bps * bin_s #maximum number of beats in a bin corresponding to 1000 bpm

            emg_signal = so.loadmat(os.path.join(ppath, recording, emg_select+'.mat'), squeeze_me=True)[emg_select]
            # emg_signal = so.loadmat(os.path.join(ppath, recording, 'LFP.mat'), squeeze_me=True)['LFP']

            # Sometimes EMG signal is inverted, if detection is poor, try flipping it
            if attempts == 1:
                emg_signal = -emg_signal
            attempts += 1

            #filter EMG
            w0 = 20 / (sr / 2)
            w1 = 150 / (sr / 2)
            emg_signal = my_bpfilter(emg_signal, w0, w1)

            #truncate to fit bin size
            emg_signal = emg_signal[:-(emg_signal.size % nbin) or None]

            #bin EMG
            EMG = emg_signal.reshape(-1, nbin)

            heartrate = []
            goodbin = []
            idxs = []
            thresholds = []

            #threshold each bin
            for i, EMGbin in enumerate(EMG):
                #The general heuristic is to progressively ramp the threshold, and then find where the number or detected peaks plateaus
                #that plateau point is assumed to be the ideal threshold for peak detection

                #determine location and prominence of all potential peaks
                idx, props = ssig.find_peaks(EMGbin, distance=min_dis_singlehb, prominence=0)
                prom = props['prominences']
                max_prom = np.max(prom)
                #Threshold from 0 to max prominence
                thr = np.arange(0, max_prom + 1, 1)

                #Peaks detected for each threshold
                # slow loop--
                # for th in thr:
                #     npeaks.append(sum(props['prominences'] >= th))
                # fast vectorization--
                npeaks = np.sum(prom[:, None] >= thr[None, :], axis=0)
                npeaks = np.array(npeaks)
                npeaks = smooth_data(npeaks, 10)

                # 1st derivative
                npeaksp = np.gradient(npeaks)
                npeaksp = smooth_data(npeaksp, 3)

                # 2nd derivative (unused)
                # npeakspp = np.gradient(npeaksp)
                # npeakspp = sleepy.smooth_data(npeakspp, 1)

                # filter to valid beat counts (corresponding to 300-1000 bpm)
                valid_indices = np.where((npeaks > min_beats) & (npeaks < max_beats))[0]

                # if no valid indices, skip this bin
                if len(valid_indices) == 0:
                    heartrate.append(np.nan)
                    goodbin.append(False)
                    thresholds.append(np.nan)
                    continue

                # Max of 1st derivative within valid range is the plateau point
                max_dx_index = valid_indices[np.argmax(npeaksp[valid_indices])]
                # min_dxx_index = np.argmin(npeakspp[max_dx_index:valid_indices[-1] + 1]) + max_dx_index

                good = True
                # Optimal threshold is assumed to be at the plateau point (where increasing threshold doesn't change peaks detected
                # indicating that we're at a point where the threshold is above noise, but below true peaks)
                iopt_thr = max_dx_index
                optimal_thr = thr[iopt_thr]

                # Detect peaks with optimal threshold, now with a minimum distance corresponding to 1000 bpm
                idx = ssig.find_peaks(EMGbin, distance=min_dis, prominence=optimal_thr)[0]
                # Compare to peaks detected without the distance constraint
                idx_nomin = ssig.find_peaks(EMGbin, distance=min_dis_singlehb,  prominence=optimal_thr)[0]

                # If removing the limit on nearness increases the heartbeats detected, then the threshold isn't good (10%))
                if (len(idx) > 0) and (len(idx_nomin) > 0):
                    if ((len(idx_nomin)-len(idx))/len(idx_nomin)) > filter_percent and len(idx_nomin) > len(idx):
                        good = False

                # Do a small local search (within 5ms) on original signal to shift peak indices (counteract filtering distortion)
                for ii in idx:
                    search_range = range(max(0, ii - 5), min(len(EMGbin), ii + 6))
                    local_max_idx = search_range[np.argmax(EMGbin[search_range])]
                    if local_max_idx != ii:
                        idx[np.where(idx == ii)[0][0]] = local_max_idx

                #Remove overlapping peaks from previous bin
                if(len(idxs)>0 and len(idx)>0):
                    if(((idx[0]+(i*nbin))-idxs[-1]) < min_dis):
                        idx = idx[1:]

                #Add to collection of peaks
                idxs.extend(idx+(i*nbin))

                #calculate heartrate between each peak
                hrates = (1/np.diff(idx))*sr*60

                #if too many gaps in peaks (drops below 300bpm), mark as bad
                if len(hrates) == 0 or sum(hrates < min_bpm)/len(hrates) > filter_percent:
                    good = False

                #average heartrate for this bin, ignoring gaps
                hr = np.mean(hrates[hrates >= min_bpm])

                #final check for reasonable heartrate
                if hr == np.nan:
                    good = False
                if hr < min_bpm or hr > max_bpm:
                    good = False

                #append to results
                heartrate.append(hr)
                goodbin.append(good)
                thresholds.append(optimal_thr)

            # percent of good bins in total recording
            quality = np.sum(goodbin) / len(goodbin)
            print('recording:', recording, 'quality:', quality)

            if quality > 0.9:
                so.savemat(hr_file,{'heartrate': heartrate, 'goodbin': goodbin, 'idxs': idxs, 'thresholds': thresholds})
                break
            if quality > prev_quality:
                print('flipped is better, keeping')
                so.savemat(hr_file,{'heartrate': heartrate, 'goodbin': goodbin, 'idxs': idxs, 'thresholds': thresholds})
                break
            prev_quality = quality

            if attempts == 1:
                print('flipping emg and trying again')
            else:
                print('flipping back, wasn\'t better')
                break
            so.savemat(hr_file, {'heartrate': heartrate, 'goodbin': goodbin, 'idxs': idxs, 'thresholds': thresholds})

    hr_data = so.loadmat(hr_file, squeeze_me=True)
    heartrate = hr_data['heartrate']
    goodbin = hr_data['goodbin']
    idxs = hr_data['idxs']
    thresholds = hr_data['thresholds']
    # print(recording, ':', np.sum(goodbin) / len(goodbin))

    return heartrate, goodbin, idxs, thresholds