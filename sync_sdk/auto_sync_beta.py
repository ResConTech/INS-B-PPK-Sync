import pandas as pd
import numpy as np
from datetime import datetime
import argparse


def clean_gnss_info_255(csv_file_path):
    """
    Remove rows where GNSS Info 1 equals 255 from INS-B CSV data.
    
    Args:
        csv_file_path (str): Path to the INS-B CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataframe with GNSS Info 1 = 255 rows removed
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Remove rows where GNSS Info 1 equals 255
    cleaned_df = df[df['GNSS Info 1'] != 255].copy()
    
    return cleaned_df


def cut_warmup_period(df):
    """
    Find the index where GNSS Info 2 switches from 20 to 28 and cut all data before that index.
    This marks the end of the warmup period and start of valid position estimates.
    
    Args:
        df (pd.DataFrame): Cleaned INS-B dataframe
        
    Returns:
        tuple: (cut_dataframe, switch_index) - DataFrame with warmup data removed and the switch index
    """
    # Find where GNSS Info 2 switches from 20 to 28
    gnss_info_2 = df['GNSS Info 2'].values
    
    # Look for the first occurrence where GNSS Info 2 equals 28 after being 20
    switch_index = None
    
    for i in range(1, len(gnss_info_2)):
        if gnss_info_2[i-1] == 20.0 and gnss_info_2[i] == 28.0:
            switch_index = i
            break
    
    if switch_index is None:
        # If no switch found, try to find first occurrence of 28
        try:
            switch_index = df[df['GNSS Info 2'] == 28.0].index[0]
        except IndexError:
            raise ValueError("No GNSS Info 2 switch from 20 to 28 found, and no value of 28 found")
    
    # Cut all data before the switch index (keep the row where it switches to 28)
    cut_df = df.iloc[switch_index:].copy()
    
    return cut_df, switch_index


def find_valid_gps_time_and_backfill(cut_df, sample_rate=0.02):
    """
    Find where GNSS Info 1 stays at 34, GNSS Info 2 stays at 28, and Sat Count stays at 9.0 
    for at least 50 samples. Use the start of that block as baseline to backfill GPS ref time.
    
    Args:
        cut_df (pd.DataFrame): DataFrame after warmup period removal
        sample_rate (float): Sample rate in seconds (default 0.02 for 50Hz)
        
    Returns:
        pd.DataFrame: DataFrame with backfilled GPS ref time
    """
    df = cut_df.copy()
    
    # Convert conditions to boolean arrays
    gnss_info_1_valid = (df['GNSS Info 1'] == 34.0)
    gnss_info_2_valid = (df['GNSS Info 2'] == 28.0)
    sat_count_valid = (df['Sat Count'] == 9.0)
    
    # Combined condition
    all_conditions_valid = gnss_info_1_valid & gnss_info_2_valid & sat_count_valid
    
    # Find consecutive blocks of valid conditions
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
            
            # Check if we have found a valid block of 50+ samples
            if current_block_length >= 50 and valid_start_idx is None:
                valid_start_idx = current_block_start
                break
        else:
            # Reset block tracking
            current_block_start = None
            current_block_length = 0
    
    if valid_start_idx is None:
        raise ValueError("No valid 50-sample block found with GNSS Info 1=34, GNSS Info 2=28, Sat Count=9.0")
    
    # Get the baseline GPS time from the start of the valid block
    baseline_gps_time = df.iloc[valid_start_idx]['GPS Time']
    
    # Backfill GPS ref time using baseline as reference point
    gps_times = df['GPS Time'].values.copy()
    milli_sample_rate = sample_rate * 1000  # Convert to milliseconds (20ms for 50Hz)
    
    # Backfill from valid_start_idx down to 0 using baseline as reference
    for i in range(valid_start_idx):
        samples_back = valid_start_idx - i
        gps_times[i] = baseline_gps_time - (samples_back * milli_sample_rate)
    
    # Update the DataFrame with backfilled GPS times
    df['GPS Time'] = gps_times
    
    return df, valid_start_idx, baseline_gps_time


def convert_ppk_time_to_milliseconds(ppk_df, num_days):
    """
    Convert PPK GPST time format to milliseconds since GPS reference week start.
    Based on find_num_milli function from reference code.
    
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
        
        days = num_days  # number of full days since beginning of GPS reference week
        day_milli = days * 24 * 3600 * 1000
        hour_milli = date_object.hour * 3600 * 1000
        minute_milli = date_object.minute * 60 * 1000
        second_milli = date_object.second * 1000
        milli_milli = date_object.microsecond // 1000
        
        total_milli = day_milli + hour_milli + minute_milli + second_milli + milli_milli
        ppk_df_copy.loc[i, 'GPST'] = total_milli
    
    return ppk_df_copy


def align_ppk_and_ins_data(cleaned_ins_df, ppk_file_path, num_days, output_decimated_ins=False, sr_ppk=0.2, sr_ins=0.02):
    """
    Align PPK and cleaned INS data using GPS reference time.
    Based on sync_data_GPS_refTime function from reference code.
    
    Args:
        cleaned_ins_df (pd.DataFrame): Cleaned INS dataframe with backfilled GPS time
        ppk_file_path (str): Path to PPK CSV file
        num_days (int): Number of full days since beginning of GPS reference week
        output_decimated_ins (bool): Whether to output decimated INS data (default False)
        sr_ppk (float): Sample rate of PPK data in seconds (default 0.2s = 5Hz)
        sr_ins (float): Sample rate of INS data in seconds (default 0.02s = 50Hz)
        
    Returns:
        tuple: (aligned_ppk_df, aligned_ins_df) - PPK aligned with full INS data
    """
    # Read and convert PPK data
    ppk_df = pd.read_csv(ppk_file_path)
    ppk_df = convert_ppk_time_to_milliseconds(ppk_df, num_days)
    
    # Get start times
    ppk_start = ppk_df['GPST'].iloc[0]
    ins_start = cleaned_ins_df['GPS Time'].iloc[0]
    
    ppk_time = np.array(ppk_df['GPST'])
    ins_time = np.array(cleaned_ins_df['GPS Time'])
    
    # Align start times
    if ppk_start < ins_start:  # PPK started before INS
        first_time_match = np.argmin(np.abs(ppk_time - ins_start))
        ppk_df = ppk_df.loc[first_time_match:, :].reset_index(drop=True)
        print(f"PPK Start adjusted: trimmed {first_time_match} samples from beginning")
    else:  # PPK started after INS
        first_time_match = np.argmin(np.abs(ins_time - ppk_start))
        cleaned_ins_df = cleaned_ins_df.loc[first_time_match:, :].reset_index(drop=True)
        print(f"INS Start adjusted: trimmed {first_time_match} samples from beginning")
    
    # Align end times based on PPK data (since it's typically shorter)
    ppk_end = ppk_df['GPST'].iloc[-1]
    ins_end_match = np.argmin(np.abs(ins_time - ppk_end))
    
    # Trim INS data to match PPK time range
    aligned_ins_df = cleaned_ins_df.loc[:ins_end_match, :].reset_index(drop=True)
    
    # Save aligned data to CSV
    ppk_df.to_csv("aligned_ppk_data.csv", index=False)
    aligned_ins_df.to_csv("aligned_ins_data_full_rate.csv", index=False)
    
    # Optionally create and save decimated INS data
    if output_decimated_ins:
        decimation_factor = int(sr_ppk / sr_ins)  # Should be 10 for 50Hz->5Hz
        
        # Find synchronized starting point by aligning with second PPK timestamp
        if len(ppk_df) > 1:
            second_ppk_time = ppk_df['GPST'].iloc[1]
            ins_times = aligned_ins_df['GPS Time'].values
            start_idx = np.argmin(np.abs(ins_times - second_ppk_time))
            # Drop the first PPK sample since we're syncing with the second one
            ppk_df_sync = ppk_df.iloc[1:].reset_index(drop=True)
        else:
            start_idx = 0  # Fallback to original behavior if only one PPK sample
            ppk_df_sync = ppk_df
        
        # Apply decimation starting from synchronized index
        decimated_ins_df = aligned_ins_df.iloc[start_idx::decimation_factor, :].reset_index(drop=True)
        
        # Ensure both have same length
        min_length = min(len(ppk_df_sync), len(decimated_ins_df))
        ppk_df_trimmed = ppk_df_sync.loc[:min_length-1, :].reset_index(drop=True)
        decimated_ins_df = decimated_ins_df.loc[:min_length-1, :].reset_index(drop=True)
        
        decimated_ins_df.to_csv("aligned_ins_data_decimated.csv", index=False)
        
        # Create time comparison DataFrame for verification
        time_comparison_df = pd.DataFrame({
            "GPST_PPK": ppk_df_trimmed['GPST'], 
            "GPS_Time_INS": decimated_ins_df['GPS Time']
        })
        time_comparison_df.to_csv("time_alignment_verification.csv", index=False)
    
    return ppk_df, aligned_ins_df


def main():
    """
    Main function to process INS-B and PPK data with command line arguments.
    """
    parser = argparse.ArgumentParser(description='Clean and synchronize INS-B and PPK data')
    
    parser.add_argument('--ins_file', '-i', required=True, 
                       help='Path to INS-B CSV file')
    parser.add_argument('--ppk_file', '-p', required=True,
                       help='Path to PPK CSV file')
    parser.add_argument('--num_days', '-d', type=int, required=True,
                       help='Number of full days since beginning of GPS reference week')
    parser.add_argument('--output_decimated', '-o', action='store_true',
                       help='Output decimated INS data (default: False)')
    parser.add_argument('--sample_rate_ins', '-sr_ins', type=float, default=0.02,
                       help='INS sample rate in seconds (default: 0.02 for 50Hz)')
    parser.add_argument('--sample_rate_ppk', '-sr_ppk', type=float, default=0.2,
                       help='PPK sample rate in seconds (default: 0.2 for 5Hz)')
    
    args = parser.parse_args()
    
    try:
        print("Step 1: Cleaning GNSS Info 1 = 255 rows...")
        cleaned_df = clean_gnss_info_255(args.ins_file)
        print(f"Removed {len(pd.read_csv(args.ins_file)) - len(cleaned_df)} rows with GNSS Info 1 = 255")
        
        print("\nStep 2: Cutting warmup period (GNSS Info 2 switch from 20 to 28)...")
        cut_df, switch_index = cut_warmup_period(cleaned_df)
        print(f"GNSS Info 2 switches from 20 to 28 at index: {switch_index}")
        print(f"Removed {switch_index} samples from warmup period")
        
        print("\nStep 3: Finding valid GPS time block and backfilling...")
        backfilled_df, valid_start_idx, baseline_gps_time = find_valid_gps_time_and_backfill(
            cut_df, args.sample_rate_ins)
        print(f"Found valid GPS time block starting at index {valid_start_idx}")
        print(f"Baseline GPS time: {baseline_gps_time}")
        
        print("\nStep 4: Aligning PPK and INS data...")
        aligned_ppk_df, aligned_ins_df = align_ppk_and_ins_data(
            backfilled_df, 
            args.ppk_file, 
            args.num_days,
            output_decimated_ins=args.output_decimated,
            sr_ppk=args.sample_rate_ppk,
            sr_ins=args.sample_rate_ins
        )
        print("Check that first PPK ref time, and first INS ref time are aligned within 5ms, if not, cut the first few samples of INS data manually. ")
        
        print(f"\nProcessing complete!")
        print(f"Final PPK data: {len(aligned_ppk_df)} samples")
        print(f"Final INS data: {len(aligned_ins_df)} samples")
        print(f"Files saved:")
        print(f"  - aligned_ppk_data.csv")
        print(f"  - aligned_ins_data_full_rate.csv")
        
        if args.output_decimated:
            print(f"  - aligned_ins_data_decimated.csv")
            print(f"  - time_alignment_verification.csv")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



# Order of operations: 

#INS-B Cleaning

#STart by cleaning 255 GNSS Info 1 rows.

# 1. We cannot use GPS ref time for warmup period because it does not stay at 0, instead we need to consider when GNSS Info 1 switches from 6 to 0, and then consider when GNSS Info 2 switches from 20 to 28. 
#This is when pos. estimate becomes valid. 

#We mark this point as the end of the warmup period. Our next task is to find when the GPS ref time starts to make sense. This point will be the start of the valid data, and we will use it to backfill the GPS ref time to the period when GNSS Info 2 switches. 



#Data flow 

#1. Clean any rows where GNSS INfo 1 is 255. 

#2. Identify the end of phase 1. This is the index where GNSS Info 2 switches from 20 to 28. Cut all samples before this index. Keep sample at GNSS Info 2 = 28. Valid Pos Est.

#3. Identify the start of phase 2. This is when GPS ref time becomes valid. Criteria for this is when GNSS Info 1 stays at 34, GNSS Info 2 stays at 28, and num satelites is 9.0 for at least 30 samples. 

#4 Go to start of that valid 30 sample block, use that as the start of the valid data. This is now "ground truth" for the GPS ref time. 

#5. Use that ground truth GPS ref time value to backfill the GPS ref time to the period when GNSS Info 2 switches. Or now the start of the valid data. 

#5.5 Convert PPK data to milliseconds from float using reference solution. 

#5.5-B Use start and end index of new cleaned INS-B Data to adjust the scope of PPK data. 

#6. Find where PPK data stops matching INS-B Data, and cut all samples after that point, cut samples before the start of valid data as well. 

#7. Return cleaned PPK data (CSV1 5hz) and cleaned INS-B data (CSV2 50hz)






