# DENGUARD AI — Streamlit Dashboard User & Operations Guide

## 1. System Overview
**DenguAlert AI** is an operational early-warning and vector-control decision support dashboard designed for municipal public health authorities in Bangladesh (Dhaka North, Dhaka South, Chattogram, Rajshahi, and Khulna City Corporations).

The dashboard translates complex meteorological patterns and multi-week epidemiological lag dynamics into actionable outbreak warnings with **14-day (2-Week)** and **28-day (4-Week)** lead times.

---

## 2. Dashboard Architecture & Module Breakdown

### 🚨 Module 1: Ward Early-Warning & Forecast
* **Primary Purpose:** Micro-level inspection for a specific municipal ward.
* **Key Components:**
  * **Dual Risk Cards:** Compares the 2-week vs. 4-week forecasts side-by-side with calibrated risk tier badges (Low, Medium, High, Critical) and corresponding public health directives.
  * **Environmental Multi-Week Indicators:** Live telemetry of current rainfall, 4-week cumulative rainfall (`rain_roll4w`), temperature bounds, relative humidity, and 1-week lagged caseload.
  * **Outbreak Trajectory Chart:** Historical case momentum (previous 16 weeks) combined with forward AI trajectory showing whether an outbreak wave is expanding or plateauing.

### 🗺️ Module 2: City-Wide Hotspot Ranking & Vector Prioritization
* **Primary Purpose:** Strategic resource allocation across all 60 monitored municipal wards.
* **Key Components:**
  * **Operational Triage Counters:** Count of wards currently classified under Critical, High, Medium, and Low threat levels.
  * **Hotspot Severity Chart:** Ranked bar graph showing the top 20 most vulnerable wards prioritized by expected 14-day case surge.
  * **Actionable Master Table:** Complete sortable directory detailing ward IDs, city corporation, dual-horizon predictions, cumulative rainfall, population density, and drainage congestion indices.

### 📈 Module 3: Temporal Trends & Weather Dynamics
* **Primary Purpose:** Epidemiological transparency and biological validation.
* **Key Components:**
  * **Dual-Axis Epidemic Waves:** Overlays multi-year weekly rainfall against recorded dengue hospitalizations to illustrate seasonal monsoon triggers.
  * **Multi-Week Lag Correlation:** Quantifies how rainfall at $t, t-1, t-2, t-3,$ and $t-4$ weeks correlates with forward transmission.
  * **Vector Thermal Suitability Scatter:** Examines the temperature-humidity envelope ($26-30^\circ\text{C}$ and $>75\%$ RH) where *Aedes aegypti* vector abundance peaks.

### 🔬 Module 4: What-If Meteorological Stress-Testing Simulator
* **Primary Purpose:** Contingency planning for extreme climate scenarios.
* **Key Components:**
  * **Interactive Perturbation Sliders:** Adjust rainfall surges ($-50\text{ mm}$ to $+150\text{ mm}$), temperature shifts ($-5^\circ\text{C}$ to $+8^\circ\text{C}$), and humidity variations ($-20\%$ to $+20\%$).
  * **Dynamic Risk Recalculation:** Demonstrates how heavy monsoon downpours compound with ward-level waterlogging indices to accelerate outbreak risk.

### 📖 Module 5: System Methodology & Benchmark Metrics
* **Primary Purpose:** Scientific governance and hackathon evaluation transparency.
* **Key Components:**
  * Full breakdown of vector breeding timelines.
  * Independent 2025 Test Set performance metrics ($N = 3,120$).
  * Calibrated threshold specification table and SHA-256 model checksums.

---

## 3. Operational Risk Tier Action Guidelines

| Risk Tier | 2W Threshold ($\hat{y}_{2w}$) | 4W Threshold ($\hat{y}_{4w}$) | Recommended Municipal Action |
| :--- | :--- | :--- | :--- |
| 🟢 **Low** | $< 0.8$ cases | $< 0.8$ cases | Routine vector surveillance; standard public awareness announcements. |
| 🟡 **Medium** | $0.8 \le \hat{y} < 4.0$ | $0.8 \le \hat{y} < 4.0$ | Intensified source reduction; clean clogged surface drains; apply targeted larvicide (*Bti*) in identified accumulation zones. |
| 🟠 **High** | $4.0 \le \hat{y} < 8.0$ | $4.0 \le \hat{y} < 8.5$ | Mobilize motorized ultra-low volume (ULV) fogging crews; issue ward-wide resident SMS alerts; preposition NS1 rapid antigen test kits at local health posts. |
| 🔴 **Critical** | $\ge 8.0$ cases | $\ge 8.5$ cases | **Immediate emergency epidemic response:** Activate municipal emergency operations center; daily targeted residual spraying; mandate hospital dengue ward surge capacity. |

---

## 4. Local Execution Guide

To launch the dashboard locally:
```bash
# Ensure virtual environment is active
streamlit run dashboard/app.py
```
Default URL: `http://localhost:8501`
