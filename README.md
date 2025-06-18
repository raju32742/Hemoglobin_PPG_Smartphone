# A Study on Non-Invasive Hemoglobin Measurement Techniques

<!-- Introduction / Background --> 
<h2 style="color:#2637a1; margin-top: 30px">1. Background</h2> 
<p style="text-align: justify; margin-top: -10px"> 
Anemia is a critical public health issue worldwide, especially in developing countries, due to the high prevalence of iron deficiency. Hemoglobin (Hb) concentration is a standard marker for diagnosing anemia, but traditional measurement methods are invasive, requiring blood draws, laboratory processing, and specialized personnel. These procedures are often painful, costly, and inaccessible for routine or large-scale screening—particularly in resource-limited settings. To address these challenges, researchers are increasingly turning toward non-invasive technologies, especially those based on optical principles such as near-infrared spectroscopy (NIRS) and photoplethysmography (PPG) [1–3]. These techniques allow hemoglobin estimation by analyzing the absorption and reflection of light through tissue, offering painless, rapid, and cost-effective alternatives [4–6]. Smartphones and custom-built LED devices now make it feasible to collect fingertip video data for PPG signal extraction, laying the groundwork for portable and accessible hemoglobin monitoring solutions [7–8]. </p>

 <!-- Add your figure image here if available --> 
 <p style="text-align:center;"> <img src="/assets/img/msc/blood.JPG" alt="Overall Functional Block Diagram of Proposed Method" style="max-width: 500px; width: 100%; border-radius: 9px; margin: 18px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.08);"> </p> <p style="text-align:center; margin-top: -30px"> <b><i>Figure: Overview of a smartphone-based non-invasive hemoglobin measurement system (adapted from [12]).</i></b> </p> 


<!-- Motivation --> 
<h2 style="color:#2637a1; margin-top: 30px">2. Motivation</h2> 
<p style="text-align: justify; margin-top: -10px"> Conventional hemoglobin testing methods are limited by their invasiveness and logistical demands. The need for simple, reliable, and noninvasive alternatives is underscored by the global burden of anemia and the growing focus on patient-centric, at-home health monitoring [1]. Recent advances in optical sensing, combined with the ubiquity of smartphones, have opened new possibilities for real-time, pain-free Hb estimation [5–8]. By leveraging PPG signals captured via consumer devices and applying advanced machine learning techniques [9–10], noninvasive hemoglobin monitoring can be democratized—making frequent self-assessment feasible for at-risk populations. The motivation behind this work is to develop and validate a low-cost, user-friendly solution that minimizes discomfort and barriers to care, supporting early diagnosis and management of anemia in both clinical and remote settings [1,11]. </p> 

<!-- Objectives --> 
<h2 style="color:#2637a1; margin-top: 30px">3. Objectives</h2> 
<p style="text-align: justify; margin-top: -10px"> 
The core objectives of this thesis are to: 
<ul>
<li>Review and compare existing non-invasive hemoglobin measurement technologies </li> 
<li>Design and develop a near-infrared (NIR) LED-based data collection board and a companion Android application for fingertip video capture</li> 
<li>Acquire a multimodal dataset, including fingertip videos and reference blood test results, from a diverse participant cohort.</li> <li>Extract and analyze optimal PPG features from recorded videos using advanced signal processing techniques</li> 
<li>Implement and evaluate several machine learning and deep learning models (e.g., LR, SVR, MLPR, DNN) for noninvasive Hb estimation</li>
</ul> </p> 

<!-- Methodology --> 
<h2 style="color:#2637a1; margin-top: 36px;">4. Methodology</h2> 
<h3 style="color:#1a237e; margin-top:28px;">4.1 System Overview</h3> 
<p style="text-align: justify; margin-top: -10px"> The system integrates a custom NIR-LED board with a smartphone camera to acquire fingertip videos under various lighting conditions [7]. The board is user-friendly, portable, and features multiple NIR LEDs to enhance signal quality. </p> 

<h3 style="color:#1a237e; margin-top:22px;">4.2 Data Collection Process</h3> 
<p style="text-align: justify; margin-top: -10px"> 
Fingertip videos were collected from subjects (aged 15–65, various genders) under three scenarios: (1) smartphone with built-in flash, (2) smartphone with NIR-LED board (no flash), and (3) reference blood sample collection. Each video was 10+ seconds (600 frames at 60 fps), with the index finger positioned on the illuminated board [8]. </p> 

<h3 style="color:#1a237e; margin-top:22px;">4.3 Video Processing & Signal Extraction</h3> 
<p style="text-align: justify; margin-top: -10px"> For each video, the central 500x500 pixel region was extracted from every frame, and the mean red channel intensity was computed to generate the raw PPG signal [4]. A fourth-order Butterworth bandpass filter was applied to suppress noise and motion artefacts [9]. The three highest-quality PPG waveforms (with prominent systolic peaks) were selected for further analysis. </p> 

<h3 style="color:#1a237e; margin-top:22px;">4.4 Feature Engineering & Selection</h3>
 <p style="text-align: justify; margin-top: -10px"> Comprehensive feature extraction was performed on each PPG cycle, including analysis of the original signal, its first and second derivatives, and frequency-domain characteristics [4,9]. Feature selection leveraged genetic algorithms to optimize the subset for model input, reducing redundancy and overfitting [10,11]. </p>
 
<h3 style="color:#1a237e; margin-top:22px;">4.5 Model Development & Validation</h3> <p style="text-align: justify; margin-top: -10px"> Multiple regression models (Linear Regression, Support Vector Regression, Multilayer Perceptron Regression, Deep Neural Networks) were trained using the selected features. K-fold (K=10) cross-validation was used for robust performance evaluation [10]. 



