from insb_gnss_sync import sync_insb_gnss_data

result = sync_insb_gnss_data(
        ins_file_path="/Users/jaymain/INSB_GNSS_Sync/CSVs/og_data/INS_07_07_2025_13_52_00_decoded.csv",
        ppk_file_path="/Users/jaymain/INSB_GNSS_Sync/CSVs/og_data/E400_SN001_processed_20250710153908.csv", 
        num_days=4,
        output_dir="/Users/jaymain/INSB_GNSS_Sync/reeeeee"
    )