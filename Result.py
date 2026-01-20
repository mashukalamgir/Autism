from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as font_manager
import pandas as pd
from numpy import mean
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def plot():#
    import numpy as np 
    import seaborn as sns  
    import matplotlib.pyplot as plt 
    #Proposed
    X=np.load("Predicted.npy")
    classes=['Autism', 'Normal']
    csfont = {'fontname':'Times New Roman'}
    cnf_matrix=confusion_matrix(X[:,0], X[:,1])  
    cm_df = pd.DataFrame(cnf_matrix,
                     index = classes,
                     columns = ['Autism', 'Normal'])
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_df,square=True, fmt='', annot=True, cmap="Blues")
    ax.set_yticklabels(classes, rotation=0, ha='right', **csfont, fontweight='bold',fontsize=12)
    ax.set_xticklabels(classes, rotation=0, ha='center', **csfont, fontweight='bold',fontsize=12)
    plt.title('Confusion Matrix', **csfont, fontweight='bold',fontsize=14)
    plt.ylabel('Actual label', **csfont, fontweight='bold',fontsize=12)
    plt.xlabel('Predicted label', **csfont, fontweight='bold',fontsize=12)    
    plt.savefig('GraphsAndImages//Confusion_Matrix.png', format="png",dpi=600)
    plt.show()
    
    FP=cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN=cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP=np.diag(cnf_matrix)
    TN=cnf_matrix.sum() - (FP + FN + TP)
    
    #Recall
    proposed_recall1=mean(TP/(TP+FN)*100)
    #Precision                              
    proposed_precision1=mean(TP/(TP+FP)*100)
    #Accuracy
    proposed_Acc1=((TP+TN)/(TP+FP+FN+TN))*100
    proposed_Acc1=mean(np.around(proposed_Acc1, 2))
    #F1 Score
    proposed_FM = (2 * proposed_recall1 * proposed_precision1)/(proposed_recall1 + proposed_precision1)
    #True Negative Rate (specificity)
    proposed_TNR = mean(TN/(TN+FP)*100)
    #True Positive Rate (sensitivity)
    proposed_FPR = mean(FP/(FP+TN))
    proposed_cohen_kappa=cohen_kappa_score(X[:,0],X[:,1],weights='quadratic')*100
    proposed_MCC = matthews_corrcoef(X[:,0],X[:,1])*100
    proposed_MSE = mean_squared_error(X[:,0],X[:,1])
    print("\nproposed Accuracy : ",proposed_Acc1)
    print("proposed Precision : ", proposed_precision1)
    print("proposed Fmeasure : ", proposed_FM)
    print("proposed Specificity : ", proposed_TNR)
    print("proposed FPR : ", proposed_FPR)
    print("proposed Recall : ", proposed_recall1)
    print("proposed cohen kappa coefficient : ", proposed_cohen_kappa)
    print("proposed MCC : ", proposed_MCC)
    print("proposed MSE : ", proposed_MSE)
    print()
    
    #ResNet
    cnf_matrix=confusion_matrix(X[:,0], X[:,2])
    FP=cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN=cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP=np.diag(cnf_matrix)
    TN=cnf_matrix.sum() - (FP + FN + TP)
    
    #Recall
    ResNet_recall1=mean(TP/(TP+FN)*100)
    #Precision                              
    ResNet_precision1=mean(TP/(TP+FP)*100)
    #Accuracy
    ResNet_Acc1=((TP+TN)/(TP+FP+FN+TN))*100
    ResNet_Acc1=mean(np.around(ResNet_Acc1, 2))
    #F1 Score
    ResNet_FM = (2 * ResNet_recall1 * ResNet_precision1)/(ResNet_recall1 + ResNet_precision1)
    #True Negative Rate (specificity)
    ResNet_TNR = mean(TN/(TN+FP)*100)
    #True Positive Rate (sensitivity)
    ResNet_FPR = mean(FP/(FP+TN))
    ResNet_cohen_kappa=cohen_kappa_score(X[:,0],X[:,2],weights='quadratic')*100
    ResNet_MCC = matthews_corrcoef(X[:,0],X[:,2])*100
    ResNet_MSE = mean_squared_error(X[:,0],X[:,2])
    print("\nResNet Accuracy : ",ResNet_Acc1)
    print("ResNet Precision : ", ResNet_precision1)
    print("ResNet Fmeasure : ", ResNet_FM)
    print("ResNet Specificity : ", ResNet_TNR)
    print("ResNet FPR : ", ResNet_FPR)
    print("ResNet Recall : ", ResNet_recall1)
    print("ResNet cohen kappa coefficient : ", ResNet_cohen_kappa)
    print("ResNet MCC : ", ResNet_MCC)
    print("ResNet MSE : ", ResNet_MSE)
    print()
    
    #AlexNet
    cnf_matrix=confusion_matrix(X[:,0], X[:,3])
    FP=cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN=cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP=np.diag(cnf_matrix)
    TN=cnf_matrix.sum() - (FP + FN + TP)
    
    #Recall
    AlexNet_recall1=mean(TP/(TP+FN)*100)
    #Precision                              
    AlexNet_precision1=mean(TP/(TP+FP)*100)
    #Accuracy
    AlexNet_Acc1=((TP+TN)/(TP+FP+FN+TN))*100
    AlexNet_Acc1=mean(np.around(AlexNet_Acc1, 2))
    #F1 Score
    AlexNet_FM = (2 * AlexNet_recall1 * AlexNet_precision1)/(AlexNet_recall1 + AlexNet_precision1)
    #True Negative Rate (specificity)
    AlexNet_TNR = mean(TN/(TN+FP)*100)
    #True Positive Rate (sensitivity)
    AlexNet_FPR = mean(FP/(FP+TN))
    AlexNet_cohen_kappa=cohen_kappa_score(X[:,0],X[:,3],weights='quadratic')*100
    AlexNet_MCC = matthews_corrcoef(X[:,0],X[:,3])*100
    AlexNet_MSE = mean_squared_error(X[:,0],X[:,3])
    print("\nAlexNet Accuracy : ",AlexNet_Acc1)
    print("AlexNet Precision : ", AlexNet_precision1)
    print("AlexNet Fmeasure : ", AlexNet_FM)
    print("AlexNet Specificity : ", AlexNet_TNR)
    print("AlexNet FPR : ", AlexNet_FPR)
    print("AlexNet Recall : ", AlexNet_recall1)
    print("AlexNet cohen kappa coefficient : ", AlexNet_cohen_kappa)
    print("AlexNet MCC : ", AlexNet_MCC)
    print("AlexNet MSE : ", AlexNet_MSE)
    print()
    
    #CNN
    cnf_matrix=confusion_matrix(X[:,0], X[:,4])
    FP=cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN=cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP=np.diag(cnf_matrix)
    TN=cnf_matrix.sum() - (FP + FN + TP)
    
    #Recall
    CNN_recall1=mean(TP/(TP+FN)*100)
    #Precision                              
    CNN_precision1=mean(TP/(TP+FP)*100)
    #Accuracy
    CNN_Acc1=((TP+TN)/(TP+FP+FN+TN))*100
    CNN_Acc1=mean(np.around(CNN_Acc1, 2))
    #F1 Score
    CNN_FM = (2 * CNN_recall1 * CNN_precision1)/(CNN_recall1 + CNN_precision1)
    #True Negative Rate (specificity)
    CNN_TNR = mean(TN/(TN+FP)*100)
    #True Positive Rate (sensitivity)
    CNN_FPR = mean(FP/(FP+TN))
    CNN_cohen_kappa=cohen_kappa_score(X[:,0],X[:,4],weights='quadratic')*100
    CNN_MCC = matthews_corrcoef(X[:,0],X[:,4])*100
    CNN_MSE = mean_squared_error(X[:,0],X[:,4])
    print("\nCNN Accuracy : ",CNN_Acc1)
    print("CNN Precision : ", CNN_precision1)
    print("CNN Fmeasure : ", CNN_FM)
    print("CNN Specificity : ", CNN_TNR)
    print("CNN FPR : ", CNN_FPR)
    print("CNN Recall : ", CNN_recall1)
    print("CNN cohen kappa coefficient : ", CNN_cohen_kappa)
    print("CNN MCC : ", CNN_MCC)
    print("CNN MSE : ", CNN_MSE)
    print()
    
    #DNN
    cnf_matrix=confusion_matrix(X[:,0], X[:,5])
    FP=cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN=cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP=np.diag(cnf_matrix)
    TN=cnf_matrix.sum() - (FP + FN + TP)
    
    #Recall
    DNN_recall1=mean(TP/(TP+FN)*100)
    #Precision                              
    DNN_precision1=mean(TP/(TP+FP)*100)
    #Accuracy
    DNN_Acc1=((TP+TN)/(TP+FP+FN+TN))*100
    DNN_Acc1=mean(np.around(DNN_Acc1, 2))
    #F1 Score
    DNN_FM = (2 * DNN_recall1 * DNN_precision1)/(DNN_recall1 + DNN_precision1)
    #True Negative Rate (specificity)
    DNN_TNR = mean(TN/(TN+FP)*100)
    #False Positive Rate 
    DNN_FPR = mean(FP/(FP+TN))
    DNN_cohen_kappa=cohen_kappa_score(X[:,0],X[:,5],weights='quadratic')*100
    DNN_MCC = matthews_corrcoef(X[:,0],X[:,5])*100
    DNN_MSE = mean_squared_error(X[:,0],X[:,5])
    print("\nDNN Accuracy : ",DNN_Acc1)
    print("DNN Precision : ", DNN_precision1)
    print("DNN Fmeasure : ", DNN_FM)
    print("DNN Specificity : ", DNN_TNR)
    print("DNN FPR : ", DNN_FPR)
    print("DNN Recall : ", DNN_recall1)
    print("DNN cohen kappa coefficient : ", DNN_cohen_kappa)
    print("DNN MCC : ", DNN_MCC)
    print("DNN MSE : ", DNN_MSE)
    print()
    
    
    from matplotlib import pyplot as plt
    import numpy as np
    #Overall performance : accuracy, precision, recall, Fmeasure, TNR comparision             
    # barWidth = 0.3     
    bars1 = [proposed_Acc1,proposed_precision1,proposed_recall1,proposed_FM,proposed_TNR]
    bars2 = [ResNet_Acc1,ResNet_precision1,ResNet_recall1,ResNet_FM,ResNet_TNR]
    bars3 = [AlexNet_Acc1,AlexNet_precision1,AlexNet_recall1,AlexNet_FM,AlexNet_TNR]
    bars4 = [CNN_Acc1,CNN_precision1,CNN_recall1,CNN_FM,CNN_TNR]
    bars5 = [DNN_Acc1,DNN_precision1,DNN_recall1,DNN_FM,DNN_TNR]
    
    plotdata = pd.DataFrame({
    "Proposed":bars1,
    "AlexNet":bars2,
    "ResNet":bars3,
    "CNN":bars4,
    "DNN":bars5
    }, 
    index=["Accuracy", "Precision", "Recall", "F-Measure", "Specificity"])
    
    plotdata.plot(kind="bar")
    plt.grid(linestyle='--', linewidth=0.3)
    plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.3)
    plt.xticks(fontsize=12, **csfont, fontweight='bold',rotation=0)
    plt.yticks(fontsize=12, **csfont, fontweight='bold',rotation=0)
    plt.title('Overall Performance', fontsize=14, **csfont, fontweight='bold')
    plt.ylim([70, 100])
    plt.legend(['Proposed','ResNet','AlexNet','CNN','DNN'], loc="lower right")
    plt.savefig('GraphsAndImages//Overall.png', format="png",dpi=600)
    plt.show()
    
    from matplotlib import pyplot as plt
    import numpy as np
    #Overall performance : accuracy, precision, recall, Fmeasure, TNR comparision             
    # barWidth = 0.3     
    bars1 = [proposed_cohen_kappa,proposed_MCC]
    bars2 = [ResNet_cohen_kappa,ResNet_MCC]
    bars3 = [AlexNet_cohen_kappa,AlexNet_MCC]
    bars4 = [CNN_cohen_kappa,CNN_MCC]
    bars5 = [DNN_cohen_kappa,DNN_MCC]
    
    plotdata = pd.DataFrame({
    "Proposed":bars1,
    "AlexNet":bars2,
    "ResNet":bars3,
    "CNN":bars4,
    "DNN":bars5
    }, 
    index=["Kappa_Score", "MCC"])
    
    plotdata.plot(kind="bar")
    plt.grid(linestyle='--', linewidth=0.3)
    plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.3)
    plt.xticks(fontsize=12, **csfont, fontweight='bold',rotation=0)
    plt.yticks(fontsize=12, **csfont, fontweight='bold',rotation=0)
    plt.title('Overall Performance', fontsize=14, **csfont, fontweight='bold')
    plt.ylim([50, 100])
    plt.legend(['Proposed','ResNet','AlexNet','CNN','DNN'], loc="center")
    plt.savefig('GraphsAndImages//MCC_kappa.png', format="png",dpi=600)
    plt.show()
    
    from matplotlib import pyplot as plt
    import numpy as np
    #Overall performance : accuracy, precision, recall, Fmeasure, TNR comparision             
    # barWidth = 0.3     
    bars1 = [proposed_FPR,proposed_MSE]
    bars2 = [ResNet_FPR,ResNet_MSE]
    bars3 = [AlexNet_FPR,AlexNet_MSE]
    bars4 = [CNN_FPR,CNN_MSE]
    bars5 = [DNN_FPR,DNN_MSE]
    
    plotdata = pd.DataFrame({
    "Proposed":bars1,
    "AlexNet":bars2,
    "ResNet":bars3,
    "CNN":bars4,
    "DNN":bars5
    }, 
    index=["FPR", "MSE"])
    
    plotdata.plot(kind="bar")
    plt.grid(linestyle='--', linewidth=0.3)
    plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.3)
    plt.xticks(fontsize=12, **csfont, fontweight='bold',rotation=0)
    plt.yticks(fontsize=12, **csfont, fontweight='bold',rotation=0)
    plt.title('Error Metrics', fontsize=14, **csfont, fontweight='bold')
    # plt.ylim([50, 100])
    plt.legend(['Proposed','ResNet','AlexNet','CNN','DNN'], loc="center")
    plt.savefig('GraphsAndImages//ErrorMetrics.png', format="png",dpi=600)
    plt.figure()
        
    #Overall ROC_AUC curve
    plt.grid(b=True, which='major', color='#666666', linestyle='-',alpha=0.3)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.3)
    auc1 = roc_auc_score(X[:,0],X[:,1]);auc1=str(auc1)
    auc2 = roc_auc_score(X[:,0],X[:,2]);auc2=str(auc2)
    auc3 = roc_auc_score(X[:,0],X[:,3]);auc3=str(auc3)
    auc4 = roc_auc_score(X[:,0],X[:,4]);auc4=str(auc4)
    auc5 = roc_auc_score(X[:,0],X[:,5]);auc5=str(auc5)
    
    fpr1, tpr1, thresholds=roc_curve(X[:,0],X[:,1])
    fpr2, tpr2, thresholds=roc_curve(X[:,0],X[:,2])
    fpr3, tpr3, thresholds=roc_curve(X[:,0],X[:,3])
    fpr4, tpr4, thresholds=roc_curve(X[:,0],X[:,4])
    fpr5, tpr5, thresholds=roc_curve(X[:,0],X[:,5])
    
    plt.plot(fpr1,tpr1,label= "Proposed_auc="+(auc1[:5]))
    plt.plot(fpr2,tpr2,label= "ResNet_auc="+(auc2[:5]))
    plt.plot(fpr3,tpr3,label= "AlexNet_auc="+(auc3[:5]))
    plt.plot(fpr4,tpr4,label= "CNN_auc="+(auc4[:5]))
    plt.plot(fpr5,tpr5,label= "DNN_auc="+(auc5[:5]))
    
    plt.title("Overall ROC Curve",fontname = "Times New Roman",fontweight='bold',fontsize=14)
    plt.xticks(fontname = "Times New Roman",fontweight='bold',fontsize=12)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=12)
    plt.ylabel("True Positive Rate",fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.xlabel("False Positive Rate",fontname = "Times New Roman",fontweight='bold',fontsize=16)
    font = font_manager.FontProperties(family='Times New Roman',style='normal',size=12,weight='bold')
    plt.legend(loc=4,prop=font)
    plt.tight_layout()
    plt.savefig('GraphsAndImages//ROC_Curve.png', format="png",dpi=600)
    plt.show()