# Ensaio Machine Learning


## Descrição

O projeto é um ensaio de diversos algoritimos de Machine Learning, seja de Classificação, Regressão ou Clusterização. Os modelos apresentados foram ajustados através dos principais hiperparamêtros de cada um, afim de melhorar as métricas de performance
e controlar situações de Underfitting e Overfitting. 

## Objetivo

Aprofundar o conhecimento sobre modelos de Machine Learning, utilizando a biblioteca scikit-learn. Aprofundar o entendimento e interpretação das métricas de performance dos algoritimos apresentados e gerar valor a partir das métricas.

## Planejamento da solução

A solução final terá 3 tabelas, apresentando a performance dos algoritimos com base em 3 conjuntos de dados diferentes, Treinamento, Validação e Teste.

## Algoritimos ensaiados

### Classificação:
Algoritmos: KNN, Decision Tree, Random Forest e Logistic Regression

Métricas de performance: Accuracy, Precision, Recall e F1-Score

### Regressão:
Algoritmos: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Polinomial Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regression Elastic Net, Polinomial Regression Lasso, Polinomial Regression Ridge e Polinomial Regression Elastic Net

Métricas de performance: R2, MSE, RMSE, MAE e MAPE

### Clusterização:
Algoritmos: K-Means e Affinity Propagation

Métricas de performance: Silhouette Score

## Ferramentas utilizadas

Bibliotecas: Python 3.11 e Scikit-learn

Algoritmos de classificação: KNN, Decision Tree, Random Forest e Logistic Regression

Algoritmos de regressão: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Polinomial Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regression Elastic Net, Polinomial Regression Lasso, Polinomial Regression Ridge e Polinomial Regression Elastic Net

Algoritmos de clusterização: K-Means e Affinity Propagation

Métricas de performance: Accuracy, Precision, Recall, F1-Score, R2, MSE, RMSE, MAE, MAPE e Silhouette Score

## Desenvolvimento
### Estratégia da solução
Para o objetivo de ensaiar os algoritmos de Machine Learning, eu vou escrever os códigos utilizando a linguagem Python, para treinar cada um dos algoritmos e vou variar seus principais parâmetros de ajuste de overfitting e observar a métrica final. O conjunto de valores que fizerem os algoritmos alcançarem a melhor performance, serão aqueles escolhidos para o treinamento final do algoritmo.

### O passo a passo
Passo 1: Divisão dos dados em treino, teste e validação.

Passo 2: Treinamento dos algoritmos com os dados de treinamento, utilizando os parâmetros “default”.

Passo 3: Medir a performance dos algoritmos treinados com o parâmetro default, utilizando o conjunto de dados de treinamento.

Passo 4: Medir a performance dos algoritmos treinados com o parâmetro “default”, utilizando o conjunto de dados de validação.

Passo 5: Testar diferentes valores de hiperparamêtros utilizando a validação cruzada para gerar as melhores métricas de performance.

Passo 6: Unir os dados de treinamento e validação

Passo 7: Retreinar o algoritmo com a união dos dados de treinamento e validação, utilizando os melhores valores para os parâmetros de controle do algoritmo.

Passo 8: Medir a performance dos algoritmos treinados com os melhores parâmetro, utilizando o conjunto de dados de teste.

Passo 9: Extrair o resultado e interpretar as métricas apresentadas

## Top 3 Insights

### Insight 1
Os algoritmos baseados em árvore apresentaram um desempenho melhor ao classificar os dados apresentados.

### Insight 2
Os algoritimos de Classificação obtiveram performances excelentes com todos os conjuntos de dados. A maior variação entre treino e teste foi de 6,20% na métrica Recall utilizando o algoritimo de DecisionTree. A menor variação foi de apenas 0,23% na métrica Recall utilizando o algoritimo LogisticRegression.

### Insight 3
Os algoritimos de regressão, DecisionTree e RandomForest, tiveram um overfitting no treinamento, 

## Resultados

![Clusterização](https://github.com/leprado06/FundML/blob/master/notebook/Clusterizacao/metrics_clus.png)

<br><br>

![Classificação](https://github.com/leprado06/FundML/blob/master/notebook/Classificacao/metrics_clas.png)

<br><br>

![Regressão](https://github.com/leprado06/FundML/blob/master/notebook/Regressao/metrics_reg.png)


