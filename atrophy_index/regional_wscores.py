"""
Compute normative w-scores for a panel of selected regions and save them to a
spreadsheet.

This reuses the normative (w-score) machinery from msa_ai.py, but instead of
collapsing regions into a single MSA-AI index, it keeps one w-score per region.

For each region a separate OLS model is fit on the reference cohort
(volume ~ 1 + Age by default), and for each scored subject the w-score is

    wscore_region = (observed - predicted) / SD_reference

Sign convention matches the MSA-AI: atrophied regions (observed below the
age/sex-expected volume) give negative w-scores.
"""

import numpy as np
import pandas as pd

from msa_ai import (msai_ai_linear, base_set, read_table,
                    rename_assemblynet_columns)


# ---------------------------------------------------------------------------
# Region panel.
#
# Each region is its own single-column feature, so the class fits one normative
# model per region and produces one w-score per region. To change which regions
# are scored, edit this list (names must match the sanitized identifier headers;
# AssemblyNet raw headers are converted automatically).
# ---------------------------------------------------------------------------
REGIONS = [
    "CerebrumTotalVolume_",
    "CerebrumWMTotalVolume_",
    "CerebrumGMTotalVolume_",
    "CerebellumTotalVolume_",
    "CerebellumWMTotalVolume_",
    "CerebellumGMTotalVolume_",
    "BrainstemVolume_",
    "AccumbensTotalVolume_",
    "AmygdalaTotalVolume_",
    "BasalForebrainTotalVolume_",
    "CaudateTotalVolume_",
    "HippocampusTotalVolume_",
    "PallidumTotalVolume_",
    "PutamenTotalVolume_",
    "ThalamusTotalVolume_",
    "VentralDCTotalVolume_",
    "Inf_LateralVentricleTotalVolume_",
    "LateralVentricleTotalVolume_",
    "x3rdVentricleVolume_",
    "x4thVentricleVolume_",
    "ExternalCSFVolume_",
]


class region_wscores(msai_ai_linear):
    """
    Per-region normative w-score calculator.

    Subclasses msai_ai_linear, building a feature_set in which each region maps
    to a single column (its own feature). Fitting and w-score computation are
    inherited unchanged; this class adds an output method that writes every
    region's w-score plus an aggregated index ("region_AI", the equal-weighted
    average of the region w-scores).

    Caveat on the aggregate: this region panel mixes brain-tissue compartments
    (which shrink with atrophy -> negative w-scores) with CSF/ventricle
    compartments (which enlarge -> positive w-scores). A plain average lets
    enlarging ventricles offset shrinking tissue, so "region_AI" is best read as
    a rough summary; the individual w-score columns are the interpretable output.
    """

    def __init__(self, reference_sheet, regions=REGIONS,
                 formula="{} ~ 1 + Age", age_lim=[40, 80]):
        # One feature per region: {feature name -> [single column]}.
        feature_set = {region: [region] for region in regions}
        self.regions = list(regions)
        # Default equal weighting; the parent normalizes weights to sum to 1, so
        # the aggregated index is the equal-weighted average of the w-scores.
        super().__init__(reference_sheet=reference_sheet,
                         feature_set=feature_set,
                         formula=formula,
                         age_lim=age_lim)

        # Also build the original 3-structure MSA-AI model (lentiform =
        # pallidum + putamen, cerebellum, brainstem) so the output can include
        # the standard MSA-AI alongside the per-region w-scores. This uses
        # msa_ai.base_set, so the lentiform w-score is computed on the summed
        # pallidum+putamen volume exactly as in msa_ai.py. To use non-equal
        # MSA-AI weights, pass weight=[...] through here.
        self._msa_ai_model = msai_ai_linear(reference_sheet=reference_sheet,
                                            feature_set=base_set,
                                            formula=formula,
                                            age_lim=age_lim)

    def score_to_spreadsheet(self, data, output_path):
        """
        Score subjects and write a spreadsheet with one w-score column per
        region, named "wscore_<Region>", plus a "wscore_lentiform" column
        (pallidum + putamen summed, from the 3-structure model) and an "MSA_AI"
        column (the standard MSA-AI: weighted average of lentiform, cerebellum,
        and brainstem w-scores).

        Whichever identifier columns exist in the input among Subject_ID, Visit,
        Age, and Sex are carried over to the output (Age and Sex are required by
        the model, so in practice they will always be present).

        Parameters
        ----------
        data : DataFrame or str
            Subjects to score, as a DataFrame or a path to a CSV/Excel file.
        output_path : str
            Destination .xlsx path.
        """
        # Accept either an in-memory DataFrame or a path to a CSV/Excel file.
        if not isinstance(data, pd.DataFrame):
            data = read_table(data)

        # Normalize any raw AssemblyNet headers so region lookups match.
        data = rename_assemblynet_columns(data)

        # compute_msa_ai (inherited) evaluates the per-region w-scores and
        # stores them on self.last_wscores_ (its aggregate return value is
        # unused here).
        self.compute_msa_ai(data)
        wscores = self.last_wscores_   # shape (subjects x regions)

        # Standard MSA-AI from the original 3-structure model (weighted average
        # of lentiform, cerebellum, brainstem w-scores). This also stores that
        # model's per-feature w-scores; feature_1 is the lentiform (pallidum +
        # putamen summed) w-score, which we surface as its own column.
        msa_ai = self._msa_ai_model.compute_msa_ai(data)
        lentiform = self._msa_ai_model.last_wscores_[:, 0]

        # Build the output columns in order: identifiers, then one w-score
        # column per region, then the lentiform w-score and the MSA-AI. Each
        # identifier column is included only if present in the input.
        out_cols = {}
        for id_col in ("Subject_ID", "Visit", "Age", "Sex"):
            if id_col in data.columns:
                out_cols[id_col] = data[id_col].to_numpy()

        for j, region in enumerate(self.regions):
            out_cols["wscore_" + region] = wscores[:, j]

        out_cols["wscore_lentiform"] = lentiform
        out_cols["MSA_AI"] = msa_ai

        out = pd.DataFrame(out_cols)
        out.to_excel(output_path, index=False)
        print("Saved region w-scores to: {}".format(output_path))
        return out


if __name__ == "__main__":
    # ----- Edit these two paths -----
    REFERENCE_SHEET = "/Users/trujillodiaz/Documents/stats/MSA_atrophy_index/data/brain_volumes_hcp_062025.xlsx"
    INPUT_FILE = "/Volumes/Paula_BM/CTR-MSA/0_Volumetric/TMSA/results/merged_report_IMSA_TMSA.csv"
    OUTPUT_FILE = "/Volumes/Paula_BM/CTR-MSA/0_Volumetric/TMSA/results/regional_wscores_msa-ai_IMSA_TMSA.xlsx"
    # --------------------------------

    model = region_wscores(reference_sheet=REFERENCE_SHEET)
    model.score_to_spreadsheet(INPUT_FILE, OUTPUT_FILE)