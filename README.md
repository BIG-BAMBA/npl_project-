# TP NLP — Modèles de langage basés sur les N-grammes

Implémentation d'un modèle de langage statistique (unigramme, bigramme, trigramme)
construit "from scratch" en Python, dans le cadre du TP NLP (Master IA & Data Engineering).

## Contenu

- `src/mini_modele_langage.py` : programme principal (menu interactif).
  Contient toutes les fonctions demandées dans le TP :
  - prétraitement du corpus (tokenisation, marqueurs `<s>`/`</s>`)
  - construction des unigrammes, bigrammes, trigrammes
  - calcul des probabilités conditionnelles (modèle bigramme)
  - lissage de Laplace
  - prédiction du mot suivant
  - génération automatique de phrase
  - calcul de la probabilité d'une phrase
  - comparaison de deux phrases
  - correction contextuelle
- `src/demo.py` : script qui exécute automatiquement toutes les questions du TP
  et affiche les résultats (pas de saisie utilisateur requise).
- `report/Rapport_TP_Ngrammes.docx` : rapport avec les réponses aux questions
  théoriques du TP.

## Utilisation

Menu interactif :

```bash
python src/mini_modele_langage.py
```

Démonstration automatique (toutes les parties du TP d'un coup) :

```bash
python src/demo.py
```

Aucune dépendance externe n'est nécessaire (bibliothèque standard Python uniquement :
`re`, `collections`).

## Corpus utilisé

```
Le chat mange du poisson.
Le chat aime le poisson.
Le chien mange de la viande.
Le chien aime la viande.
Le chat joue dans le jardin.
Le chien joue dans le jardin.
```

## Auteur

TP réalisé dans le cadre du cours de Traitement Automatique du Langage Naturel (NLP),
Institut Supérieur d'Informatique — Département Intelligence Artificielle et Data Engineering.
