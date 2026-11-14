"""SecondSite - Interactive Gradio UI for Clinical Triage."""
import gradio as gr
import pandas as pd
from models import predict_patient

CP_MAP = {"1: Typical Angina": 1, "2: Atypical Angina": 2, "3: Non-Anginal Pain": 3, "4: Asymptomatic": 4}
SEX_MAP = {"Female (0)": 0, "Male (1)": 1}
FBS_MAP = {"<= 120 mg/dL (0)": 0, "> 120 mg/dL (1)": 1}
ECG_MAP = {"0: Normal": 0, "1: ST-T Wave Abnormality": 1, "2: Left Ventricular Hypertrophy": 2}

def triage_interface(age, sex_label, cp_label, trestbps, chol, fbs_label, restecg_label):
    patient = {
        "age": float(age),
        "sex": SEX_MAP[sex_label],
        "cp": CP_MAP[cp_label],
        "trestbps": float(trestbps) if trestbps > 0 else None,
        "chol": float(chol) if chol > 0 else None,
        "fbs": FBS_MAP[fbs_label],
        "restecg": ECG_MAP[restecg_label],
    }
    
    result = predict_patient(patient)
    prob = result["risk"]
    decision = result["decision"]
    rule_out = result["rule_out_line"]
    refer = result["refer_line"]
    
    if decision == "Routine follow-up":
        color = "#16a34a" # Green
        action = f"Calibrated risk ({prob:.1%}) is below the demographic rule-out line ({rule_out:.1%}). No immediate stress test required."
    elif decision == "Unsure - order exercise stress test":
        color = "#ca8a04" # Yellow
        action = f"Calibrated risk ({prob:.1%}) falls inside the uncertainty band ({rule_out:.1%} - {refer:.1%}). Order an exercise stress test before allocating an angiography slot."
    else:
        color = "#dc2626" # Red
        action = f"Calibrated risk ({prob:.1%}) exceeds the referral precision threshold ({refer:.1%}). Prioritize for specialist evaluation."

    sex_str = "Female" if patient["sex"] == 0 else "Male"
    
    md = f"""
    <div style="padding: 20px; border-radius: 10px; border: 2px solid {color}; background-color: {color}10;">
        <h2 style="color: {color}; margin-top: 0;">Triage Decision: {decision}</h2>
        <ul style="font-size: 16px;">
            <li><b>Calibrated Coronary Risk:</b> {prob:.1%}</li>
            <li><b>Recommended Action:</b> {action}</li>
            <li><b>Demographic Guardrail ({sex_str}):</b> Rule-out threshold automatically adjusted to {rule_out:.1%} to maintain >= 90% demographic sensitivity.</li>
        </ul>
        <p style="font-size: 12px; color: #64748b; margin-bottom: 0;"><i>{result['disclaimer']}</i></p>
    </div>
    """
    return md

with gr.Blocks(title="SecondSite — District Chest-Pain Triage", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# SecondSite: Routine Checkup Cardiovascular Triage
        **User:** Outpatient District Chest-Pain Clinic | **Inputs:** 7 Routine Checkup Vitals Only (No upfront stress-test required)
        
        *Adjust the vitals below to see how the model's calibrated risk and triage decision adapt dynamically using demographic-fair thresholds.*"""
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            age = gr.Slider(25, 85, value=58, step=1, label="Age (years)")
            sex = gr.Radio(list(SEX_MAP.keys()), value="Female (0)", label="Biological Sex")
            cp = gr.Dropdown(list(CP_MAP.keys()), value="4: Asymptomatic", label="Chest Pain Type (cp)")
            trestbps = gr.Slider(80, 210, value=140, step=1, label="Resting Blood Pressure (mmHg, 0 = Missing)")
            chol = gr.Slider(0, 500, value=260, step=1, label="Serum Cholesterol (mg/dL, 0 = Missing)")
            fbs = gr.Radio(list(FBS_MAP.keys()), value="<= 120 mg/dL (0)", label="Fasting Blood Sugar")
            restecg = gr.Dropdown(list(ECG_MAP.keys()), value="0: Normal", label="Resting ECG")
            
        with gr.Column(scale=1):
            out_md = gr.HTML()
            
    # Auto-update when sliders/dropdowns change for the "What-If" simulator feel
    inputs = [age, sex, cp, trestbps, chol, fbs, restecg]
    for inp in inputs:
        inp.change(fn=triage_interface, inputs=inputs, outputs=out_md)
        
    demo.load(fn=triage_interface, inputs=inputs, outputs=out_md)

    gr.Markdown("### Clinical Edge-Case Presets")
    gr.Examples(
        examples=[
            [43, "Female (0)", "4: Asymptomatic", 132, 341, "> 120 mg/dL (1)", "2: Left Ventricular Hypertrophy"], 
            [59, "Male (1)", "4: Asymptomatic", 138, 271, "<= 120 mg/dL (0)", "2: Left Ventricular Hypertrophy"], 
            [58, "Male (1)", "4: Asymptomatic", 145, 0, "<= 120 mg/dL (0)", "0: Normal"], 
        ],
        inputs=inputs,
        label="Click to load a patient profile"
    )

if __name__ == "__main__":
    demo.launch(share=True)
