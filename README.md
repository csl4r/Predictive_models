# Predictive_models
This repository includes a central Mods.R file and several data sets. The data was sourced from the Parkinson’s Progressive Markers Initiative (PPMI : http://www.ppmi-info.org/) . Data was filtered to include only complete cases across 14 different non-motor and biomarker features. 
 
The code and data were for two binary predictive probability analyses: classification of early Parkinson’s disease (PD) versus controls and classification of early PD versus SWEDD (scans without evidence of dopamine deficit). Binary logistic regression, general additive (GAM), decision tree, random forest and XGBoost models were fitted using non-motor clinical and biomarker features. You will need install the relevant libraries (i.e. for XGBoost, GAM, randomForest, and rpart). The required libraries are as follows:
library(car); library (data.table); library(DMwR); library(grid); library(psych); library(QuantPsyc); library(corrplot) library(tidyr); library(MASS); library(pscl); library(ROSE); library(rpart); library(dplyr); library(ggplot2); library(effects); library(randomForest)
library(caret); library(rpart.plot);  library(rattle); library(pROC); library(xgboost)

Data sets (csv format)
Note, simply following the Mods.R script ensures correct usage of the data sets. Some data sets  have long and unfortunately convoluted names. Of course, you can rename any data set and insert the changed name into the model (e.g. mod1<- model(outcome ~. , data = “new_Data_name”).

The hcpd.dat and pdsw2.dat sets include the early PD/control and early PD/SWEDD data respectively. Both data sets were split into train and test sets using random stratified partitioning. The main early PD/control data sets are train1 and test1. The main early PD/SWEDD data sets  are train50a and test50a.  All 5 models were trained on features determined by model-based feature selection from train1 in the case of the early PD/control binary classification. For the early PD/SWEDD classification, all 5 models were trained on features determined by model-based feature selection using train50a. In both analyses model features producing the highest AUC were adopted final model features.  The final models were then tested on the validation data. 

All 5 models were applied to validation set data:  test1 for early PD/controls and test50a for early PD/SWEDD.  This was a cross-validation paradigm, where models were tested on data unseen by the models during training.

1. “datfinalpatnoMay3checkrevUp.csv” = all the final filtered data for the early PD, control and SWEDD groups

Early PD/Control data

2. "hcpd_dat.csv" = all early PD/control data (does not include SWEDD data); 130 HC, 295 (early) PD

3. “train1.csv” = early PD/control training data set (91 HC, 207 early PD); after data partition: 97 HC, 207 PD
Model specific feature sets selected from train1 (early PD/control):
-GLM (logistic regression): “LR1_rsCNSTrbdMocaQ3.csv”, includes 6 features (7 features actually given MoCA converted to quartiles); note MoCA was covered to quartiles as it violated Box-Tidwell linearity of the logit; "LR1_rsMocaDummies.csv”: this the same as the latter but converts MoCA to dummy values
-GAM also used the  “LR1_rsCNSTrbdMocaQ3.csv” feature set (6 features) but as it is not affected by non-linearity between a predictor and the legit of outcome, the numeric version of MoCA was used along with other 5 features (see Mods.R for details)
-Decision tree: “treeNative6.csv”; 6 features 
-Random forest: “Natrf1.csv” ; 6 features
-XGBoost: "xgbTrain2.csv”; 11 features; must be converted to matrix format; dependent variable should be removed and retained separately.

4. “test1.csv” = validation data set early PD/control (39 HC, 88 PD); features here unseen by models

5.  “hcSWa_pd4SW_July9.csv” = validation test set for two best early PD/control classifiers (GAM and XGBoost); latter two models applied to this data set to assess ability of models to predict conversion from SWEDD to PD; note early PD class from models map to PD in this test set (DV is catpd4SW), but PD here is actually the SWEDD category; 39 HC, 43 PD

___________________________________________________________________________
Early PD/SWEDD data

6. “pdsw2.csv” = early PD/SWEDD data set (does not include controls);  295 PD, 43 SWEDD. The training data set partition was “train50a.csv”

7. "smote_pdswJune14.csv”: (early PD/SWEDD) SMOTE subsampled data (44 PD; 44 SWEDD) derived from “train50a.csv”. All models except the decision tree used this SMOTE data set; SMOTE data models had higher AUC compared to non-subsampled data. The decision tree used 
used a different SMOTE data set: “TreeSub_resampSm88_66.csv” (PD 88, SWEDD 66). This was from SMOTE subsampling  data obtained during resampling, which proved to has the highest AUC for the decision tree. 

8. Model-based feature selection, with final feature set determined by features contributing to model with highest AUC: 
-GLM: "glm2dat.csv" ; includes years of education, which violated Box-Tidwell linearity of logit, converted to quartiles (and as dummy variables); 6 features (8 counting education years quartiles as dummies)
-GAM: "glmDat.csv" : same variables (6) as GLM but quartile or dummy form of years of education on not needed.
-Decision tree: "tree2dat.csv”; 10 features; again, derived from “TreeSub_resampSm88_66.csv”, which in turn was derived from Train50a
-Random forest: “rf2.csv”; 12 features 
-XGBoost: “"xgtrSM.csv”: features must be converted to matrix format: all 14 features used

9. “test50a.csv” (147 early PD, 21 SWEDD); validation or test set (not altered by SMOTE subsampling); features here unseen by models

10. The two curated data sets can be matched with model early PD/control GAM and XGBoost model outcomes and early PD/SWEDD model outcomes to arrive at the percentage of model predictions coinciding the curated (12-36 month) diagnoses. 
“cure1_reducedSetAug15.csv”: PPMI curated data: “datscanValsImAug15datScanPlusVisInterp.csv”: SPECT data. 

Brief on results
 All five models achieved >.80 AUC cross-validated (CV) accuracy to distinguish early PD from controls using non-motor clinical and biomarker features.  Classifier performance, across models, was consistently lower in the early PD/SWEDD analyses. In both early PD/control and early PD/SWEDD analyses, and across all models, hyposmia was the single most important feature to classification; RDBQ held the next most common high rank of importance. Alpha-synuclein was a feature of import to early PD/control but not early PD/SWEDD classification and the Epworth Sleepiness scale was antithetically important to the latter but not former. 

Corresponding author: charlie9@yorku.ca; cslfalcon@gmail.com
