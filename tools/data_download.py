from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PhysicalAI-SmartSpaces",  #
    repo_type="dataset",                      #
    allow_patterns="MTMC_Tracking_2025/*",   #
    ignore_patterns=".",            #
    local_dir="dataset"           #
)