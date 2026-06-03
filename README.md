# Hotel Booking Demand — Predição de Cancelamentos de Reservas

**Atividade Avaliativa Final — Inteligência Computacional**  
Faculdade de Tecnologia de Jundiaí (FATEC) | Curso Superior de Tecnologia em Ciência de Dados  
**Professor:** Me. Mateus Guilherme Fuini **Alunos: Allan Oliveira, Gabriel Passos, Guilherme Cavalcante**

---

## Descrição

Projeto de classificação binária que prediz se uma reserva de hotel será **cancelada** (`is_canceled = 1`) ou **mantida** (`is_canceled = 0`), utilizando o dataset público [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

- **119.390 registros** | **32 atributos originais**
- Dois tipos de hotel: Resort Hotel e City Hotel

---

## Estrutura do Notebook

| Seção | Conteúdo |
|---|---|
| 1. Importações | Bibliotecas utilizadas |
| 2. Carregamento dos Dados | Leitura do CSV via URL, visão geral e valores ausentes |
| 3. EDA | Distribuição do alvo, histogramas, boxplots, heatmap de correlação, análise temporal |
| 4. Pré-processamento | Remoção de colunas com data leakage, engenharia de atributos, split treino/val/teste |
| 5. Pipeline | `ColumnTransformer` com imputação, escalonamento e One-Hot Encoding |
| 6. Regressão Logística | Treinamento e avaliação no conjunto de validação |
| 7. Cross-Validation | 5-Fold Stratified K-Fold sobre o conjunto de treino |
| 8. GridSearchCV (KNN) | Otimização de hiperparâmetros sobre amostra de 20 mil registros |
| 9. Avaliação Final | Métricas no conjunto de teste (nunca visto) + comparativo LR vs KNN |
| 10. Discussão | Dificuldades, limitações e possíveis melhorias |

---

## Resultados

| Modelo | Accuracy Validação | CV Média (5-fold) | Accuracy Teste |
|---|---|---|---|
| Regressão Logística | ~81.5% | 81.5% ± 0.23% | ~81.3% |
| KNN (best config.) | — | — | — |

> **Melhor configuração KNN:** `n_neighbors=11`, `weights='distance'`, `metric='manhattan'`

---

## Como Executar

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd <pasta>

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Abra o notebook
jupyter notebook P2_IC.ipynb
```

O dataset é carregado automaticamente via URL pública (TidyTuesday / GitHub), sem necessidade de download manual.

---

## Dependências

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Engenharia de Atributos

Três novos atributos criados a partir das colunas originais:

| Atributo | Derivação |
|---|---|
| `total_nights` | `stays_in_week_nights + stays_in_weekend_nights` |
| `total_guests` | `adults + children + babies` |
| `room_changed` | `reserved_room_type != assigned_room_type` (binário) |
