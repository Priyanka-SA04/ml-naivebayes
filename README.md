# Bloom's Taxonomy Level Classification using Naive Bayes

## Theoretical Foundation

### Bloom's Taxonomy Framework
The classifier predicts one of six cognitive levels from Bloom's revised taxonomy:

| Level | Name        | Description                          | Example Keywords          |
|-------|-------------|--------------------------------------|---------------------------|
| 1     | Remembering | Recall facts and basic concepts      | define, list, recall      |
| 2     | Understanding | Explain ideas or concepts           | summarize, paraphrase     |
| 3     | Applying    | Use information in new situations    | implement, solve          |
| 4     | Analyzing   | Draw connections among ideas         | compare, contrast         |
| 5     | Evaluating  | Justify a stand or decision          | critique, defend          |
| 6     | Creating    | Produce new or original work         | design, construct         |

### Naive Bayes Mathematics
The classifier uses Bayes' Theorem:

Where:
- `P(y|X)`: Posterior probability of class y given features X
- `P(X|y)`: Likelihood of features given class y
- `P(y)`: Prior probability of class y
- `P(X)`: Marginal probability of features

For text classification, we use the multinomial variant:

### Key Features
The model analyzes these question characteristics:
- **Lexical Features**:
  - Bloom's verbs (e.g., "compare" → Analyzing)
  - Question length and word frequency
- **Syntactic Features**:
  - Question structure (interrogatives, complexity)
  - Presence of comparative phrases
- **Semantic Features**:
  - Cognitive demand indicators
  - Domain-specific terminology

### Model Characteristics
| Aspect          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| Algorithm Type  | Probabilistic classifier                                                    |
| Training Speed  | Fast (single pass through data)                                             |
| Data Efficiency | Works well even with limited training examples                              |
| Interpretability| High (clear feature probabilities)                                          |



