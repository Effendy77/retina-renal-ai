# Task 3: Time-to-ESRD Survival

## ⚠️ Primary Analysis Code Location

The full, published survival analysis pipeline (including KFRE comparison, figures, and 5-fold CV scripts) is maintained in a dedicated repository:

🔗 **[Effendy77/retina-esrd-survival](https://github.com/Effendy77/retina-esrd-survival)**

## Development Note

This directory is reserved for utility scripts or minimal survival task integration with the main `src.train` structure.
If you need to quickly test the survival module within this codebase, use the following arguments:
`--task survival` with `--target time` and `--event_col event`.
