"""
vrpipeline.io
=============
Real I/O for the raw modalities used by both source papers:

  - Multi-channel iEEG/sEEG/ECoG recordings (EDF, HDF5/.mat-h5, or a plain
    (T, N) float array + channel-name list you already loaded)
  - Electrode coordinates + anatomical labels (CSV: id,x,y,z,label)
  - Post-resection binary segmentation masks (NIfTI, co-registered to the
    same space as the electrode coordinates -- see Kini et al. Fig. 1
    imaging pipeline: ANTs-registered pre/post-resection MRI + ITK-SNAP
    semi-automatic segmentation. That registration/segmentation step is a
    neuro-imaging task done in ANTs/ITK-SNAP/FreeSurfer outside Python; this
    module consumes its output mask.)
  - Clinical seizure annotations (EEC / UEO / END, semiology) as in
    Supplementary Table S1 of Kini et al.

No network fetching happens here -- you point these loaders at files you
have already downloaded (e.g. from IEEG.org / OpenNeuro, matching the
`Study012`, `HUP*` naming used in the source papers).
"""
from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np


@dataclasses.dataclass
class IEEGRecording:
    """Container for one continuous iEEG/sEEG/ECoG clip.

    Attributes
    ----------
    data : ndarray, shape (T, N)
        Time x channel matrix, raw voltage (uV) or already-preprocessed.
    fs : float
        Sampling frequency (Hz).
    channels : list[str]
        Channel/electrode labels, length N, in the same order as columns
        of `data`.
    eec_sample, ueo_sample, end_sample : Optional[int]
        Sample indices of earliest electrographic change, unequivocal
        electrographic onset, and seizure end, if this clip is a seizure
        (None for interictal clips). Matches Table S1 columns C/D/E.
    """
    data: np.ndarray
    fs: float
    channels: list
    eec_sample: Optional[int] = None
    ueo_sample: Optional[int] = None
    end_sample: Optional[int] = None

    def pre_seizure_window(self) -> "IEEGRecording":
        """Return the baseline pre-ictal window: duration = END-EEC,
        ending at EEC (Kini et al., Materials & Methods)."""
        if self.eec_sample is None or self.end_sample is None:
            raise ValueError("EEC/END not set on this recording")
        dur = self.end_sample - self.eec_sample
        start = max(0, self.eec_sample - dur)
        return IEEGRecording(self.data[start:self.eec_sample], self.fs,
                              self.channels)

    def seizure_window(self) -> "IEEGRecording":
        if self.ueo_sample is None or self.end_sample is None:
            raise ValueError("UEO/END not set on this recording")
        return IEEGRecording(self.data[self.ueo_sample:self.end_sample],
                              self.fs, self.channels)


def load_array(data: np.ndarray, fs: float, channels: list,
               eec_s: Optional[int] = None, ueo_s: Optional[int] = None,
               end_s: Optional[int] = None) -> IEEGRecording:
    """Wrap an already-loaded (T, N) numpy array (e.g. from your own EDF/MEF
    reader such as MNE, pyedflib, or ieeg.org's `ieeg-python` client)."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be (T, N)")
    if data.shape[1] != len(channels):
        raise ValueError("channels length must match data.shape[1]")
    return IEEGRecording(data, float(fs), list(channels), eec_s, ueo_s, end_s)


def load_edf(path: str, channels: Optional[list] = None) -> IEEGRecording:
    """Load an EDF/EDF+ file via MNE (real, no mock). Requires `mne`."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    if channels is not None:
        raw = raw.pick_channels(channels)
    data = raw.get_data().T  # (T, N)
    return IEEGRecording(data, float(raw.info["sfreq"]), list(raw.ch_names))


def load_h5_ieeg(path: str, data_key: str = "evData",
                  fs_key: str = "Fs", chan_key: str = "channels"
                  ) -> IEEGRecording:
    """Load the HDF5 clip format used by IEEG.org exports (as consumed by
    the original EpiVR `virtual_resection` module: evData/Fs/channels)."""
    import h5py
    with h5py.File(path, "r") as f:
        data = np.asarray(f[data_key])
        if data.ndim == 2 and data.shape[0] < data.shape[1]:
            data = data.T
        fs = float(np.asarray(f[fs_key]).flat[0])
        chans = []
        for ref in np.asarray(f[chan_key]).ravel():
            obj = f[ref] if isinstance(ref, h5py.Reference) else ref
            arr = np.asarray(obj).ravel()
            chans.append("".join(chr(int(c)) for c in arr))
    return IEEGRecording(data, fs, chans)


def load_electrode_coordinates(csv_path: str) -> dict:
    """Load electrode coordinates + labels.

    Expected CSV columns (no header, matches EpiVR ELECTRODE_LABELS format):
        electrode_id, x, y, z, label

    Returns
    -------
    dict: label -> (x, y, z) float tuple
    """
    coords = {}
    with open(csv_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            _id, x, y, z, label = parts[:5]
            coords[label.strip()] = (float(x), float(y), float(z))
    return coords


def load_resection_mask(nifti_path: str):
    """Load a co-registered post-resection binary segmentation mask.

    Returns (volume: ndarray, affine: 4x4 ndarray) via nibabel.
    Threshold matches EpiVR (probability > 0.2 -> resected), replicated
    exactly from `util_virtual_resection.get_resected_electrodes`.
    """
    import nibabel as nib
    img = nib.load(nifti_path)
    vol = np.asarray(img.get_fdata())
    vol = (vol > 0.2).astype(np.uint8)
    return vol, img.affine


@dataclasses.dataclass
class SeizureAnnotation:
    patient_id: str
    seizure_num: int
    eec: float
    ueo: float
    end: float
    semiology: str  # FAS, FIA, FBTC, BTC, or ''


def load_annotation_table(csv_path: str) -> list:
    """Load a Table-S1-style clinical seizure annotation CSV with columns
    patient_id,seizure_num,eec,ueo,end,semiology (write this yourself from
    the annotations you download per-patient from IEEG.org; Table S1 in
    the supplementary PDF gives the exact values reported for the original
    28-patient cohort if you want to reproduce that cohort specifically)."""
    out = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(SeizureAnnotation(
                patient_id=row["patient_id"],
                seizure_num=int(row["seizure_num"]),
                eec=float(row["eec"]),
                ueo=float(row["ueo"]),
                end=float(row["end"]),
                semiology=row.get("semiology", "").strip(),
            ))
    return out
