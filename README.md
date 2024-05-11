<h1 align="center">Financial Transactions Anomaly Detection</h1>
  <h3 align="center">Detecting Anomalies in Financial Transaction using Deep Autoencoder Networks.</h3>

</div>

<br/>

<div align="center">
  <a href="#"><img alt="My Github" src="https://img.shields.io/badge/Still%20being%20fixed!-8A2BE2"></a>
  <a href="https://github.com/AndrewKim2807"><img alt="My Github" src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white"></a>
  <a href="https://github.com/AndrewKim2807/Financial-Anomaly-Detection"><img alt="License" src="https://img.shields.io/badge/License-MIT-red"></a>
  <a href="#"><img alt="Visual Studio Code" src="https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?logo=visual-studio-code&logoColor=white"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff"></a>
</div>

<br/>



![Thumbnail](https://github.com/AndrewKim2807/Financial-Anomaly-Detection/blob/main/Live%20Anomaly%20Detection%20in%20Financial%20Transactions.png)

The Association of Certified Fraud Examiners, a renowned organization dedicated to the study and prevention of fraudulent activities, has reported in its comprehensive Global Fraud Study of the year 2016 that on average, a standard organization experiences a financial loss equivalent to 5% of its yearly revenues as a result of fraudulent behaviors.

According to the esteemed Joseph T. Wells, a highly regarded authority in the field of forensic accounting and fraud examination, the term "fraud" can be defined and understood as:
```"the abuse of one's occupation for personal enrichment through the deliberate misuse of an organization's resources or assets."```

A similar more recent study conducted by the auditors of PwC, revealed that 30% of the study respondents experienced losses of between 100,000 and 5 million USD in the last 24 months. The study also showed that financial statement fraud cause by far the greatest median loss of the surveyed fraud schemes. Organizations are simultaneously expediting the digitization and restructuring of business processes, which notably impact Accounting Information Systems (AIS) and more broadly, Enterprise Resource Planning (ERP) systems.

![Figure1](https://github.com/AndrewKim2807/Financial-Anomaly-Detection/blob/main/figure%201.png)

The visualization above shows a Hierarchical perspective of an Accounting Information System (AIS) encompasses discrete tiers of abstraction, specifically the data related to business processes, the financial data, and (3) the detailed technical entries stored in specified database tables. Gradually, these systems accumulate extensive amounts of digital evidence at a nearly atomic scale. This is especially true for the diary entries of a company documented in its main ledger and subsidiary accounts. SAP, a leading provider of ERP software, approximates that around **76%** of the global transaction revenue is processed through one of their systems.

## Key Features
1. Classification of Financial Anomalies
2. Deep Learning Based Methodology
3. Autoencoder Neural Network (AENN)
4. Anomaly Detection
5. Comprehensive Dataset
6. Fraud Detection Models [Confidential]
7. Combination of Predictions

## Libraries
1. **Pandas**
2. **NumPy**
3. **Matplotlib** & **Seaborn**
4. **Scikit-learn**
5. **TensorFlow** & **Keras**

## Autoencoder Neural Networks (AENNs)
The purpose of this section is to acquaint ourselves with the fundamental idea and principles behind constructing a deep autoencoder neural network (AENN). We will discuss the key components and the precise network architecture of AENNs, along with a demonstration of its implementation using the open-source machine learning library PyTorch.

below illustrates a schematic view of an autoencoder network comprised of two non-linear mappings (fully connected feed forward neural networks) referred to as encoder $f_\\theta: \\mathbb{R}^{dx} \\mapsto \\mathbb{R}^{dz}$ and decoder $g_\\theta: \\mathbb{R}^{dz} \\mapsto \\mathbb{R}^{dx}$.

![Architecture](https://github.com/AndrewKim2807/Financial-Anomaly-Detection/blob/main/NN%20Architecture.png)

Furthermore, AENNs can be interpreted as \"lossy\" data **compression algorithms**. They are \"lossy\" in a sense that the reconstructed outputs will be degraded compared to the original inputs. The difference between the original input $x^i$ and its reconstruction $\\hat{x}^i$ is referred to as **reconstruction error**. In general, AENNs encompass three major building blocks:
> 1. an encoding function $f_\\theta$
> 2. a decoding mapping function $g_\\theta$
> 3. and a lost function $\\mathcal{L_{\\theta}}$

Most commonly the encoder and the decoder mapping functions consist of **several layers of neurons followed by a non-linear function** and shared parameters $\\theta$. The encoder mapping $f_\\theta(\\cdot)$ maps an input vector (e.g. an \"one-hot\" encoded transaction) $x^i$ to a compressed representation $z^i$ referred to as latent space $Z$. This hidden representation $z^i$ is then mapped back by the decoder $g_\\theta(\\cdot)$ to a re-constructed vector $\\hat{x}^i$ of the original input space (e.g. the re-constructed encoded transaction). Formally, the nonlinear mappings of the encoder- and the decoder-function can be defined by:

$f_\\theta(x^i) = s(Wx^i + b)$, and $g_\\theta(z^i) = s′(W′z^i + d)$


where $s$ and $s′$ denote non-linear activations with model parameters $\\theta = \\{W, b, W', d\\}$, $W \\in \\mathbb{R}^{d_x \\times d_z}, W' \\in \\mathbb{R}^{d_z \\times d_y}$ are weight matrices and $b \\in \\mathbb{R}^{dx}$, $d \\in \\mathbb{R}^{dz}$ are offset bias vectors.

