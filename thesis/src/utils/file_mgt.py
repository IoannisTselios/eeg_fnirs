import random
import os
from pathlib import Path


import random
from pathlib import Path

def get_random_eeg_file_paths(extension: str, count: int = None) -> list[Path]:
    """
    Get EEG files with a given extension from all subdirectories beyond the fixed root path.

    Parameters
    ----------
    extension : str
        The file extension to look for ("xdf", "fif", "snirf").
    count : int, optional
        The number of file paths to return. If None, return all files.

    Returns
    -------
    list[Path]
        A list of EEG file paths.
    """

    # Fixed root directory
    root_path = Path("L:\\LovbeskyttetMapper\\CONNECT-ME\\CONMED3\\Dataoptagelser\\NIRS-EEG\\")

    # Validate extension
    if extension not in ["xdf", "fif", "snirf"]:
        raise ValueError(f"Invalid extension '{extension}'. Must be one of: 'xdf', 'fif', 'snirf'.")

    # Ensure the base directory exists
    if not root_path.is_dir():
        raise FileNotFoundError(f"Directory '{root_path}' does not exist.")

    # Find all matching files recursively in all subdirectories
    paths = list(root_path.rglob(f"*.{extension}"))

    # If count is specified and smaller than total files, return a random subset
    if count is not None and count < len(paths):
        return random.sample(paths, count)
    
    return paths  # Return all files if count is None or more than available files


def get_random_eeg_file_paths_grouped_by_session(extension: str, session_count: int = 1) -> list[list[str]]:
    """
    Get random sets of EEG files making up one session.
    
    Parameters
    ----------
    extension: str
        The file extension to look for ("xdf" or "fif").
    session_count: int
        The number of sessions to return.
    """
    
    if session_count < 1 or extension not in ["xdf", "fif"]:
        raise ValueError("Invalid parameters")
    
    paths = list()

    main_path = os.path.join("data", "raw")

    for drug_folder in [f.path for f in os.scandir(main_path) if f.is_dir()]:

        for session_folder in [f.path for f in os.scandir(drug_folder) if f.is_dir()]:

            temp_paths = list()

            for path in Path(session_folder).rglob("*." + extension):
                temp_paths.append(path)
            paths.append(temp_paths)

    if session_count >= len(paths):
        return paths

    return random.sample(paths, session_count)