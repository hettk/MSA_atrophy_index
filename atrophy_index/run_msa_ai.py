from msa_ai import msai_ai_linear

model = msai_ai_linear(reference_sheet="/Users/trujillodiaz/Documents/stats/MSA_atrophy_index/data/brain_volumes_hcp_062025.xlsx")   # your real reference file
model.score_to_spreadsheet(
    "/Volumes/Paula_BM/CTR-MSA/0_Volumetric/TMSA/results/merged_report_TMSA_IMSA.csv",
    "/Volumes/Paula_BM/CTR-MSA/0_Volumetric/TMSA/results/wscores_msa-ai_TMSA_IMSA.xlsx",
)