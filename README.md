# A Multi-Step Multivariate LSTM Model-based Melt Tank Product Quality  Prediction System

This repository contains source code and experiment results from the paper entitled 'A Multi-Step Multivariate LSTM Model-based Melt Tank Product Quality Prediction System'

**Abstract**

_Product quality is consistently the top concern of consumers, so that it determines the success of businesses. Product quality is constituted and specified by multiple factors in product manufacturing factories, from the selection of production materials, operating machinery to the production and preservation of products to bring to consumers. In these stages, the mixing of production materials in melt tanks plays an essential role in determining the quality of output products of the production line. In this paper, we introduce an approach based on the Multi-step multivariate Long-short term memory (LSTM) model to predict the mixing quality of raw materials in melt tanks in food manufacturing plants. By choosing multivariable and multi-step input, we present how to choose the parameters to predict the product quality from the melt tank. Appling the approach, the accuracy was improved from 69% to 78% for predicting quality of product in a melt tank from a real factory in Korea. This approach benefits the operators, engineers to control and operate machinery to produce good quality products_


Some experiment results of the system.

~~![](Evaluation%20results/systemUI.png)
System user interface

![](Evaluation%20results/1-MELT_TEMP-INSP.png)

Trained model with 1 input-step, features: MELT_TEMP, INSP


![](Evaluation%20results/1-MELT_TEMP-MOTORSPEED.png)

**Trained model with 1 input-step:**

![](Evaluation%20results/1-MELT_TEMP-INSP.png)

![](Evaluation%20results/1-MELT_TEMP-MOTORSPEED.png)

![](Evaluation%20results/1-MELT_TEMP-MOTORSPEED-INSP.png)

![](Evaluation%20results/1-MOTORSPEED-INSP.png)


**Trained model with 5 input-steps:**

![](Evaluation%20results/5-MELT_TEMP-INSP.png)

![](Evaluation%20results/5-MELT_TEMP-MOTORSPEED.png)

![](Evaluation%20results/5-MELT_TEMP-MOTORSPEED-INSP.png)

![](Evaluation%20results/5-MOTORSPEED-INSP.png)


**Trained model with 10 input-steps:**

![](Evaluation%20results/10-MELT_TEMP-INSP.png)

![](Evaluation%20results/10-MELT_TEMP-MOTORSPEED.png)

![](Evaluation%20results/10-MELT_TEMP-MOTORSPEED-INSP.png)

![](Evaluation%20results/10-MOTORSPEED-INSP.png)
~~
See Evalutaion results folder for more detail.

**Comparison**

![](Evaluation%20results/comparison_chart.png)


If you need any further information related to the source code, don't hestitate to contact me at: thuhuyen(at)kgu.ac.kr




