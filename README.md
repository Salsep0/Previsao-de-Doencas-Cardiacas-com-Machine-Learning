# Previsão de Doenças Cardíacas usando Machine Learning

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Descrição do Projeto

Este projeto tem como objetivo **prever a probabilidade de um paciente ter uma doença cardíaca** com base em variáveis clínicas, demográficas e de estilo de vida. Utilizando técnicas de **aprendizado supervisionado**, foram aplicados e comparados dois modelos de classificação:

- **Regressão Logística** — modelo interpretável e eficiente para identificar padrões lineares
- **Random Forest Classifier** — modelo robusto e não linear, ideal para capturar relações complexas entre variáveis

## 🎯 Objetivos

- Realizar análise exploratória de dados (EDA) para compreender a distribuição das variáveis
- Desenvolver e comparar modelos de machine learning para classificação
- Identificar os fatores mais relevantes para prever doenças cardíacas
- Avaliar o desempenho dos modelos usando métricas apropriadas

## 📊 Dataset

O dataset contém informações de pacientes com as seguintes características:

- **Variáveis clínicas**: pressão arterial, colesterol, dor no peito, etc.
- **Variáveis demográficas**: idade, sexo
- **Variáveis de estilo de vida**: tabagismo

## 🛠️ Tecnologias Utilizadas

- **Python 3.7+**
- **Pandas & NumPy** - Manipulação de dados
- **Matplotlib, Seaborn & Plotly** - Visualização de dados
- **Scikit-learn** - Machine Learning
- **Jupyter Notebook** - Ambiente de desenvolvimento

## 📈 Resultados

### Desempenho dos Modelos

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| Regressão Logística | 0.65 | 0.63 | 0.61 | 0.61  |
| Random Forest | 0.62 | 0.60 | 0.87 | 0.57 | 0.56 |

### Variáveis Mais Importantes (Random Forest)
1. **thalach** - Frequência cardíaca máxima
2. **oldpeak** - Depressão do segmento ST
3. **age** - Idade  
4. **cp** - Tipo de dor no peito
5. **chol** - Colesterol

## 🤝 Contribuição

Contribuições são bem-vindas! Siga os passos:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autor

Paulo a
- GitHub: [@Salsep0](https://github.com/Salsep0)
- LinkedIn: [Seu Perfil](https://www.linkedin.com/in/paulo-vitor-83a095225)

## 🙏 Agradecimentos

- Dataset: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Comunidade de Ciência de Dados



## 📁 Estrutura do Projeto
heart-disease-prediction/
│
├── data/
│ └── heart_disease_dataset.csv
│
├── notebooks/
│ └── heart_disease_analysis.ipynb
│
├── src/
│ ├── init.py
│ ├── data_preprocessing.py
│ └── model_training.py
│
├── models/
│ ├── logistic_regression_model.pkl
│ └── random_forest_model.pkl
│
├── requirements.txt
├── environment.yml
└── README.md