"""
INSB-GNSS Synchronization SDK

A simple SDK for synchronizing INS-B and PPK GNSS data with temporal alignment.
Provides a single function interface for easy integration.

Usage:
    from insb_gnss_sync import sync_insb_gnss_data
    
    result = sync_insb_gnss_data(
        ins_file_path="path/to/ins_data.csv",
        ppk_file_path="path/to/ppk_data.csv", 
        num_days=5,
        output_dir="./output"
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Tuple, Dict, Optional


def clean_gnss_info_255(csv_file_path: str) -> pd.DataFrame:
    """
    Remove rows where GNSS Info 1 equals 255 from INS-B CSV data.
    
    Args:
        csv_file_path (str): Path to the INS-B CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataframe with GNSS Info 1 = 255 rows removed
    """
    df = pd.read_csv(csv_file_path)
    cleaned_df = df[df['GNSS Info 1'] != 255].copy()
    return cleaned_df


def cut_warmup_period(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Find the index where GNSS Info 2 switches from 20 to 28 and cut all data before that index.
    This marks the end of the warmup period and start of valid position estimates.
    
    Args:
        df (pd.DataFrame): Cleaned INS-B dataframe
        
    Returns:
        tuple: (cut_dataframe, switch_index) - DataFrame with warmup data removed and the switch index
    """
    gnss_info_2 = df['GNSS Info 2'].values
    switch_index = None
    
    for i in range(1, len(gnss_info_2)):
        if gnss_info_2[i-1] == 20.0 and gnss_info_2[i] == 28.0:
            switch_index = i
            break
    
    if switch_index is None:
        try:
            switch_index = df[df['GNSS Info 2'] == 28.0].index[0]
        except IndexError:
            raise ValueError("No GNSS Info 2 switch from 20 to 28 found, and no value of 28 found")
    
    cut_df = df.iloc[switch_index:].copy()
    return cut_df, switch_index


def find_valid_gps_time_and_backfill(cut_df: pd.DataFrame, sample_rate: float = 0.02) -> Tuple[pd.DataFrame, int, float]:
    """
    Find where GNSS Info 1 stays at 34, GNSS Info 2 stays at 28, and Sat Count stays at 9.0 
    for at least 50 samples. Use the start of that block as baseline to backfill GPS ref time.
    
    Args:
        cut_df (pd.DataFrame): DataFrame after warmup period removal
        sample_rate (float): Sample rate in seconds (default 0.02 for 50Hz)
        
    Returns:
        tuple: (DataFrame with backfilled GPS ref time, valid_start_idx, baseline_gps_time)
    """
    df = cut_df.copy()
    
    gnss_info_1_valid = (df['GNSS Info 1'] == 34.0)
    gnss_info_2_valid = (df['GNSS Info 2'] == 28.0)
    sat_count_valid = (df['Sat Count'] == 9.0)
    all_conditions_valid = gnss_info_1_valid & gnss_info_2_valid & sat_count_valid
    
    valid_start_idx = None
    current_block_start = None
    current_block_length = 0
    
    for i, is_valid in enumerate(all_conditions_valid):
        if is_valid:
            if current_block_start is None:
                current_block_start = i
                current_block_length = 1
            else:
                current_block_length += 1
            
            if current_block_length >= 50 and valid_start_idx is None:
                valid_start_idx = current_block_start
                break
        else:
            current_block_start = None
            current_block_length = 0
    
    if valid_start_idx is None:
        raise ValueError("No valid 50-sample block found with GNSS Info 1=34, GNSS Info 2=28, Sat Count=9.0")
    
    baseline_gps_time = df.iloc[valid_start_idx]['GPS Time']
    gps_times = df['GPS Time'].values.copy()
    milli_sample_rate = sample_rate * 1000
    
    for i in range(valid_start_idx):
        samples_back = valid_start_idx - i
        gps_times[i] = baseline_gps_time - (samples_back * milli_sample_rate)
    
    df['GPS Time'] = gps_times
    return df, valid_start_idx, baseline_gps_time


def convert_ppk_time_to_milliseconds(ppk_df: pd.DataFrame, num_days: int) -> pd.DataFrame:
    """
    Convert PPK GPST time format to milliseconds since GPS reference week start.
    
    Args:
        ppk_df (pd.DataFrame): PPK dataframe with GPST column
        num_days (int): Number of full days since beginning of GPS reference week
        
    Returns:
        pd.DataFrame: PPK dataframe with GPST converted to milliseconds
    """
    ppk_df_copy = ppk_df.copy()
    
    for i, time_str in ppk_df_copy['GPST'].items():
        date_string = time_str
        date_format = "%H:%M:%S.%f"
        date_object = datetime.strptime(date_string, date_format)
        
        days = num_days
        day_milli = days * 24 * 3600 * 1000
        hour_milli = date_object.hour * 3600 * 1000
        minute_milli = date_object.minute * 60 * 1000
        second_milli = date_object.second * 1000
        milli_milli = date_object.microsecond // 1000
        
        total_milli = day_milli + hour_milli + minute_milli + second_milli + milli_milli
        ppk_df_copy.loc[i, 'GPST'] = total_milli
    
    return ppk_df_copy


def trim_ppk_to_ins_end(ppk_df: pd.DataFrame, ins_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim PPK data to match the end time of INS data.
    
    Args:
        ppk_df (pd.DataFrame): PPK dataframe with GPST timestamps
        ins_df (pd.DataFrame): INS dataframe with GPS Time timestamps
        
    Returns:
        pd.DataFrame: Trimmed PPK dataframe
    """
    ins_end_time = ins_df['GPS Time'].iloc[-1]
    ppk_times = ppk_df['GPST'].values
    
    # Find PPK samples that are within INS recording time
    valid_ppk_mask = ppk_times <= ins_end_time
    trimmed_ppk_df = ppk_df[valid_ppk_mask].reset_index(drop=True)
    
    return trimmed_ppk_df


def align_ppk_and_ins_data(cleaned_ins_df: pd.DataFrame, ppk_df: pd.DataFrame, 
                          output_decimated_ins: bool = False, sr_ppk: float = 0.2, 
                          sr_ins: float = 0.02) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align PPK and cleaned INS data using GPS reference time.
    
    Args:
        cleaned_ins_df (pd.DataFrame): Cleaned INS dataframe with backfilled GPS time
        ppk_df (pd.DataFrame): PPK dataframe with converted timestamps
        output_decimated_ins (bool): Whether to output decimated INS data (default False)
        sr_ppk (float): Sample rate of PPK data in seconds (default 0.2s = 5Hz)
        sr_ins (float): Sample rate of INS data in seconds (default 0.02s = 50Hz)
        
    Returns:
        tuple: (aligned_ppk_df, aligned_ins_df) - Aligned datasets
    """
    ppk_start = ppk_df['GPST'].iloc[0]
    ins_start = cleaned_ins_df['GPS Time'].iloc[0]
    
    ppk_time = np.array(ppk_df['GPST'])
    ins_time = np.array(cleaned_ins_df['GPS Time'])
    
    # Align start times
    if ppk_start < ins_start:
        first_time_match = np.argmin(np.abs(ppk_time - ins_start))
        ppk_df = ppk_df.loc[first_time_match:, :].reset_index(drop=True)
    else:
        first_time_match = np.argmin(np.abs(ins_time - ppk_start))
        cleaned_ins_df = cleaned_ins_df.loc[first_time_match:, :].reset_index(drop=True)
    
    # Trim PPK data to not exceed INS end time (keep full INS data)
    ppk_df = trim_ppk_to_ins_end(ppk_df, cleaned_ins_df)
    
    # Return both datasets - PPK trimmed to INS range, INS data kept full
    return ppk_df, cleaned_ins_df


def create_decimated_data(ppk_df: pd.DataFrame, aligned_ins_df: pd.DataFrame, 
                         sr_ppk: float = 0.2, sr_ins: float = 0.02) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create decimated INS data synchronized with PPK timestamps.
    
    Args:
        ppk_df (pd.DataFrame): Aligned PPK dataframe
        aligned_ins_df (pd.DataFrame): Aligned full-rate INS dataframe
        sr_ppk (float): Sample rate of PPK data in seconds
        sr_ins (float): Sample rate of INS data in seconds
        
    Returns:
        tuple: (ppk_df_sync, decimated_ins_df) - Synchronized decimated datasets
    """
    decimation_factor = int(sr_ppk / sr_ins)
    
    # Find synchronized starting point by aligning with second PPK timestamp
    if len(ppk_df) > 1:
        second_ppk_time = ppk_df['GPST'].iloc[1]
        ins_times = aligned_ins_df['GPS Time'].values
        start_idx = np.argmin(np.abs(ins_times - second_ppk_time))
        # Drop the first PPK sample since we're syncing with the second one
        ppk_df_sync = ppk_df.iloc[1:].reset_index(drop=True)
    else:
        start_idx = 0
        ppk_df_sync = ppk_df
    
    # Apply decimation starting from synchronized index
    decimated_ins_df = aligned_ins_df.iloc[start_idx::decimation_factor, :].reset_index(drop=True)
    
    # Ensure both have same length
    min_length = min(len(ppk_df_sync), len(decimated_ins_df))
    ppk_df_sync = ppk_df_sync.loc[:min_length-1, :].reset_index(drop=True)
    decimated_ins_df = decimated_ins_df.loc[:min_length-1, :].reset_index(drop=True)
    
    return ppk_df_sync, decimated_ins_df


def sync_insb_gnss_data(ins_file_path: str, ppk_file_path: str, num_days: int,
                       output_dir: str = "./output", output_decimated: bool = True,
                       sample_rate_ins: float = 0.02, sample_rate_ppk: float = 0.2) -> Dict[str, str]:
    """
    Complete INSB-GNSS data synchronization pipeline.
    
    Args:
        ins_file_path (str): Path to INS-B CSV file
        ppk_file_path (str): Path to PPK CSV file  
        num_days (int): Number of full days since beginning of GPS reference week
        output_dir (str): Output directory for synchronized files (default: "./output")
        output_decimated (bool): Whether to output decimated INS data (default: True)
        sample_rate_ins (float): INS sample rate in seconds (default: 0.02 for 50Hz)
        sample_rate_ppk (float): PPK sample rate in seconds (default: 0.2 for 5Hz)
        
    Returns:
        Dict[str, str]: Dictionary with paths to output files
        
    Raises:
        ValueError: If data processing fails or invalid parameters provided
        FileNotFoundError: If input files don't exist
    """
    # Validate inputs
    if not os.path.exists(ins_file_path):
        raise FileNotFoundError(f"INS file not found: {ins_file_path}")
    if not os.path.exists(ppk_file_path):
        raise FileNotFoundError(f"PPK file not found: {ppk_file_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Clean GNSS Info 1 = 255 rows
        cleaned_df = clean_gnss_info_255(ins_file_path)
        
        # Step 2: Cut warmup period
        cut_df, switch_index = cut_warmup_period(cleaned_df)
        
        # Step 3: Find valid GPS time and backfill
        backfilled_df, valid_start_idx, baseline_gps_time = find_valid_gps_time_and_backfill(
            cut_df, sample_rate_ins)
        
        # Step 4: Process PPK data
        ppk_df = pd.read_csv(ppk_file_path)
        ppk_df = convert_ppk_time_to_milliseconds(ppk_df, num_days)
        
        # Step 5: Align datasets
        aligned_ppk_df, aligned_ins_df = align_ppk_and_ins_data(
            backfilled_df, ppk_df, output_decimated, sample_rate_ppk, sample_rate_ins)
        
        # Prepare output file paths
        output_files = {}
        
        # Save full-rate aligned data
        ppk_output_path = os.path.join(output_dir, "aligned_ppk_data.csv")
        ins_full_output_path = os.path.join(output_dir, "aligned_ins_data_full_rate.csv")
        
        aligned_ppk_df.to_csv(ppk_output_path, index=False)
        aligned_ins_df.to_csv(ins_full_output_path, index=False)
        
        output_files["ppk_aligned"] = ppk_output_path
        output_files["ins_full_rate"] = ins_full_output_path
        
        # Create decimated data if requested
        if output_decimated:
            ppk_df_sync, decimated_ins_df = create_decimated_data(
                aligned_ppk_df, aligned_ins_df, sample_rate_ppk, sample_rate_ins)
            
            # Save decimated data
            ins_decimated_path = os.path.join(output_dir, "aligned_ins_data_decimated.csv")
            verification_path = os.path.join(output_dir, "time_alignment_verification.csv")
            
            decimated_ins_df.to_csv(ins_decimated_path, index=False)
            
            # Create time comparison for verification
            time_comparison_df = pd.DataFrame({
                "GPST_PPK": ppk_df_sync['GPST'], 
                "GPS_Time_INS": decimated_ins_df['GPS Time']
            })
            time_comparison_df.to_csv(verification_path, index=False)
            
            output_files["ins_decimated"] = ins_decimated_path
            output_files["verification"] = verification_path
        
        return output_files
        
    except Exception as e:
        raise ValueError(f"Data synchronization failed: {str(e)}")


# Convenience function for backward compatibility
def process_insb_gnss(ins_file: str, ppk_file: str, num_days: int, output_dir: str = "./output"):
    """
    Simplified interface for basic usage.
    """
    return sync_insb_gnss_data(ins_file, ppk_file, num_days, output_dir)