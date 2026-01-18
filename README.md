# 📊 Prédiction du non-renouvellement des contrats de Location Longue Durée (LLD)

## 🚗 Contexte du projet

La **Location Longue Durée (LLD)** est une solution de financement très utilisée par les entreprises pour la gestion de leurs véhicules professionnels.  
Un enjeu majeur pour les acteurs du secteur est **d’anticiper les contrats susceptibles de ne pas être renouvelés**, afin de mettre en place des actions de fidélisation ciblées et proactives.

Ce projet vise à **développer un modèle de Machine Learning** et une **application décisionnelle** permettant d’identifier les clients à risque de non-renouvellement.

> 📌 Projet réalisé **de manière indépendante** dans le cadre d’un **Projet de Fin d’Études – Master 2 Data**.

---

## 🎯 Objectifs du projet

- 🔍 Détecter les contrats susceptibles de ne pas être renouvelés
- 📈 Fournir un **score de risque clair et interprétable**
- 🤝 Aider les équipes **commerciales et juridiques** à prioriser leurs actions
- 🖥️ Proposer une **application interactive** simple d’utilisation

---

## 🗂️ Données utilisées

- 📄 Environ **40 000 contrats anonymisés**
- 📌 Données issues d’un **système de gestion de contrats**
- 🔐 Données **totalement anonymisées** (conformité RGPD)

### Exemples de variables :
- Ancienneté du contrat
- Montant mensuel
- Nombre de prestations incluses
- Retards / restitutions tardives
- Services associés (assurance, carburant, etc.)
- Statut du contrat (actif / clôturé)

---

## 🔎 Analyse Exploratoire des Données (EDA)

L’analyse exploratoire a permis de :

- Comprendre la répartition des contrats renouvelés vs non-renouvelés
- Identifier les variables les plus corrélées au non-renouvellement
- Nettoyer les données :
  - Suppression des doublons
  - Traitement des valeurs aberrantes
  - Gestion des valeurs manquantes
- Préparer les données pour la modélisation

**Observations clés :**
- Les clients non-renouvelés présentent souvent :
  - Une restitution tardive
  - Moins de prestations incluses
  - Moins de services complémentaires

---

## 🤖 Modélisation et Machine Learning

### Modèles testés
Trois algorithmes ont été comparés :

| Modèle | Recall | F1-score |
|------|-------|---------|
| Régression Logistique | 0.64 | 0.63 |
| Random Forest | 0.72 | 0.71 |
| **XGBoost** | **0.75** | **0.73** |

### Modèle retenu
👉 **XGBoost**, car il offre le meilleur compromis entre :
- Détection des clients à risque (Recall)
- Performance globale (F1-score)
- Robustesse sur données déséquilibrées

---

## 📊 Résultats du modèle (XGBoost)

### Matrice de confusion
La matrice de confusion permet de visualiser :
- Les contrats correctement prédits
- Les erreurs de prédiction (faux positifs / faux négatifs)

➡️ Le modèle identifie correctement **environ 7 clients à risque sur 10**.

### Courbe ROC
- **AUC ROC = 0.82**
- Le modèle distingue efficacement les clients à risque des clients non à risque
- Performance nettement supérieure à un modèle aléatoire

### Taux global de bonne prédiction
- Environ **71 %** de prédictions correctes
- Suffisant pour un usage **opérationnel et décisionnel**

---

## 🖥️ Application Streamlit

Une application interactive a été développée avec **Streamlit**.

### Fonctionnalités principales :
- 📂 Import de données anonymisées (Excel)
- 📊 Analyse exploratoire interactive
- ⚖️ Comparaison de plusieurs modèles
- 📈 Visualisation des performances (ROC, métriques)
- 🚨 Détection automatique des clients à risque
- 🏆 Classement des **Top 50 clients les plus à risque**
- 📥 Export des résultats en Excel

---

## 🧱 Architecture du projet

```bash
├── app.py
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── eda.py
│   ├── comparaison_models.py
│   ├── training_xgboost.py
│   ├── Courbe_ROC.py
│   └── predict.py
├── models/
│   └── xgboost_model.joblib
├── outputs/
│   └── figures/
├── assets/
│   └── logo_bpce.jpg
├── requirements.txt
└── README.md
