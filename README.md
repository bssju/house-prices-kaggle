# House Prices — Advanced Regression Techniques

Solução para a competição [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) do Kaggle.

---

## Resultado

| Métrica | Valor |
|---|---|
| RMSLE (Kaggle) | **0.12949** |
| Algoritmo | LightGBM |
| Features | 236 (após encoding) |

---

## Estrutura

```
house-prices-kaggle/
├── house_prices_julianaburato.ipynb
├── README.md
└── submission.csv
```

---

## Etapas do Projeto

### 1. Análise Exploratória (EDA)

- Distribuição do `SalePrice` — skew positivo corrigido com transformação `log1p`
- Correlação das features numéricas com o target (`OverallQual` lidera com folga)
- Mapeamento de valores ausentes por coluna

### 2. Tratamento de NaN

NaN neste dataset raramente indica dado faltante — na maioria dos casos indica ausência da feature:

| Grupo | Estratégia |
|---|---|
| Categóricas de ausência (`PoolQC`, `GarageType`, ...) | `'None'` |
| Numéricas de ausência (`GarageArea`, `TotalBsmtSF`, ...) | `0` |
| `LotFrontage` | Mediana por bairro |
| Demais categóricas | Moda |

### 3. Encoding Ordinal

Variáveis com hierarquia natural mapeadas para inteiros preservando a ordem qualitativa (`None = 0`):

| Colunas | Escala |
|---|---|
| `ExterQual`, `BsmtQual`, `KitchenQual`, `FireplaceQu`, ... | Po=1, Fa=2, TA=3, Gd=4, Ex=5 |
| `BsmtFinType1`, `BsmtFinType2` | Unf=1, LwQ=2, Rec=3, BLQ=4, ALQ=5, GLQ=6 |
| `GarageFinish` | Unf=1, RFn=2, Fin=3 |

### 4. Feature Engineering

| Feature | Definição | Justificativa |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | Compradores avaliam área total |
| `TotalBath` | `FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath` | Convenção do mercado imobiliário |
| `HouseAge` | `YrSold − YearBuilt` | Captura depreciação |
| `YearsSinceRemod` | `YrSold − YearRemodAdd` | Captura valorização por atualização |
| `WasRemodeled` | `YearRemodAdd != YearBuilt` | Flag binário de reforma |
| `HasGarage` / `HasBasement` / `HasFireplace` | `feature > 0` | Presença/ausência tem impacto não-linear |

### 5. Modelo: LightGBM

#### Arquitetura

Gradient Boosting com crescimento por folha (leaf-wise), treinando árvores sequencialmente onde cada árvore corrige os erros da anterior.

```
Dados (236 features)
    → Árvore 1 → resíduo 1
    → Árvore 2 → corrige resíduo 1
    → ...
    → Árvore N → predição final = soma de todas as contribuições
```

#### Treinamento

- Optimizer: Gradient Boosting com `learning_rate=0.05`
- `num_leaves=63` — controla a complexidade de cada árvore
- `min_child_samples=20` — regularização por mínimo de amostras por folha
- GridSearchCV com 5-fold CV para seleção de hiperparâmetros

### 6. Avaliação

- RMSLE no Kaggle: **0.12949**
- Cross-validation (5 folds): `0.1244 ± 0.0046`

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
---

*Juliana Burato — 2026*
