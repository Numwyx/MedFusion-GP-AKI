# MedFusion-GP-AKI: External Cohort Dataset and Pseudocode

This repository contains the processed and de-identified dataset used for **external validation** of the MedFusion-GP-AKI model, developed for predicting acute kidney injury (AKI) risk in traumatic spinal cord injury (TSCI) patients.  
The dataset and accompanying pseudocode are provided to promote transparency, reproducibility, and FAIR (Findable, Accessible, Interoperable, Reusable) data sharing.

---

## 📂 Repository Contents
| File | Description |
|------|--------------|
| `test1_rounded.csv` | Rounded dataset (two-decimal precision) used for external model validation |
| `data_dictionary.csv` | Detailed schema with data types, example values, English descriptions, and units |
| `pseudocode.md` | Minimal pseudocode describing how AUC, AP, Brier Score, CITL, slope, and DCA were computed |
| `README.md` | Repository documentation (this file) |
| `LICENSE` | Licensing information (CC BY 4.0 for data, MIT for code) |

---

## 🧠 Dataset Overview
- **Population:** Traumatic spinal cord injury (TSCI) patients  
- **Scope:** 24-hour ICU admission period  
- **Sample unit:** Individual patient record  
- **Format:** UTF-8 CSV file with numeric and categorical features  
- **Continuous values:** Rounded to **two decimal places**  
- **Categorical encoding:** Integer-coded for interoperability  

---

## 📊 Variable Description

| Variable | Type | Unit / Category | Description |
|-----------|------|-----------------|--------------|
| **GCS** | Continuous | points | Glasgow Coma Scale score used to assess level of consciousness |
| **CCI** | Continuous | points | Charlson Comorbidity Index indicating chronic disease burden |
| **HR** | Continuous | beats/min | Heart rate measured in beats per minute |
| **MBP** | Continuous | mmHg | Mean blood pressure reflecting average arterial pressure |
| **Temp** | Continuous | °C | Body temperature measured at admission |
| **WBC** | Continuous | ×10⁹/L | White blood cell count |
| **HCO3-** | Continuous | mmol/L | Plasma bicarbonate concentration |
| **Potassium** | Continuous | mmol/L | Serum potassium concentration |
| **Lactate** | Continuous | mmol/L | Serum lactate concentration |
| **TSCI-Level** | Categorical | 1 = Upper cervical, 2 = Lower cervical, 3 = Upper thoracic, 4 = Lower thoracic | Encoded spinal cord injury level |
| **VP-Use** | Categorical | 0 = No, 1 = Yes | Vasopressor use within the first 24 hours |

---

## ⚙️ How to Use
You may use this dataset to:
- Benchmark predictive models against the MedFusion-GP-AKI architecture.  
- Reproduce discrimination and calibration metrics (AUC, AP, Brier, CITL, Slope).  
- Perform external or transfer validation of machine learning models on TSCI-related AKI outcomes.  

Refer to **`pseudocode.md`** for step-by-step instructions on metric computation.

---

## 📜 License
- **Data:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)  
- **Code / Pseudocode:** MIT License  

Users are free to share, adapt, and build upon this work with proper attribution.

---

## 📚 Citation
If you use this dataset, please cite the following:

> Tang Y., Li Y., Zhou B., *et al.* (2025). *MedFusion-GP-AKI: A Generalizable Gaussian Process Model for AKI Prediction in TSCI Patients.*  
> GitHub: [https://github.com/Numwyx/MedFusion-GP-AKI](https://github.com/Numwyx/MedFusion-GP-AKI)  
> DOI (Zenodo): `10.xxxx/zenodo.xxxxxx` (to be added after Zenodo release)

---

## 📬 Contact
**Corresponding author:** Yuqian Li  
**Affiliation:** Department of Anesthesiology, The First Affiliated Hospital of Xinjiang Medical University  
**Address:** 393 Xinyi Road, Xinshi District, Urumqi 830054, Xinjiang, China  
**Tel:** +86 0991-4362851  
**E-mail:** [liyuqian@xjmu.edu.cn](mailto:liyuqian@xjmu.edu.cn)
