# Summary
Hardware (stl files for 3D printing) and Software (code for analyzing, visualizing, and UI) for my senior project titled "Insect Olfaction-Based Odor Detection System".  
Insect used here is 'drosophila', which is also known 'fruit fly'.

# Hardware
There are 'stl files for 3D Printing' folder for Hardware settings.  
- Version1 was for making system box, which is used in experimental process of the project.  
- Version2 is for making system box, which is going to be used in practical process of the project.

# Software
I edited a little bit of 'tracktor' library, and then used it.

### Codes for Analyzing
- Behavior_Analysis_AllDay.py
  - It analyzes all data files in output folder.
  - And accumulated result will be saved in the 'output_accumulated' folder to be analyzed later with whole data.
- BA_Responding_Pattern_standard.py & BA_Responding_Pattern_detail.py
  - BA means BenzAldehyde
  - ~_standard.py visualizes Paths of flies with only dots.
  - ~_detail.py visualizes Paths of flies in more detail by using dots, lines, numbers, and colormap.

### Code for UI demo
- UI_demo.py
  - It is for the demo version of UI, which will be used for practical field test of this whole system.
  - You can see how it works breifly with "UI demonstration video.mp4". 

# 'output' folder structure
- output
  - ...
  - 20220119
  - 20220120 (= an example saved here)
    - Day
      - Cartridge1_W1118_W1118_W1118_W1118_W1118_Or7a_D_DPG_1-heptanol_2-pentanol_1AA_BA
        - detail_result
          - (results of analyzing) 
        - standard_result
          - (results of analyzing)
        - t20220120_182854.csv
        - t20220120_182854.mp4
        - (results of analyzing)
      - (results of analyzing)
    - Night
      - ... 
  - ... 
- output_accumulated
  - (results of analyzing)


Other files in the 'output' folder will appear as results of the codes for analyzing.
(Files already in the 'output' folder are from experiments.)

# Reference:
Vivek Hari Sridhar. (2017). vivekhsridhar/tracktor: Tracktor (tracktor). Zenodo. https://doi.org/10.5281/zenodo.1134016
