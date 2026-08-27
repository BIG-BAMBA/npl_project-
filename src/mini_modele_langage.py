# -*- coding: utf-8 -*-
"""
Mini modèle de langage statistique basé sur les N-grammes
TP - NLP - Modèles de langage basés sur les N-grammes
"""

import re
from collections import Counter

# =========================================================
# PARTIE 3 - CORPUS DE TRAVAIL
# =========================================================

CORPUS_BRUT = [
    "Le chat mange du poisson.",
    "Le chat aime le poisson.",
    "Le chien mange de la viande.",
    "Le chien aime la viande.",
    "Le chat joue dans le jardin.",
    "Le chien joue dans le jardin.",
]

# Corpus supplémentaire pour la Partie 8 (correction contextuelle)
CORPUS_CORRECTION = [
    "Il a sept ans.",
    "Elle a sept ans.",
    "Mon frère a sept ans.",
    "Il a cet objet.",
    "Elle a cet objet.",
    "Il prend cet objet.",
]


# =========================================================
# PARTIE 1 - PRÉTRAITEMENT DU CORPUS
# =========================================================

def tokeniser(phrase):
    """Met en minuscules, enlève la ponctuation, découpe en tokens,
    ajoute les marqueurs <s> et </s>."""
    phrase = phrase.lower()
    phrase = re.sub(r"[^\w\s<>/]", "", phrase)  # garde les lettres/chiffres/espaces
    tokens = phrase.split()
    return ["<s>"] + tokens + ["</s>"]


def construire_corpus_tokenise(corpus_brut):
    return [tokeniser(p) for p in corpus_brut]


def construire_vocabulaire(corpus_tokenise):
    vocab = set()
    for phrase in corpus_tokenise:
        vocab.update(phrase)
    return vocab


def nombre_total_tokens(corpus_tokenise):
    return sum(len(phrase) for phrase in corpus_tokenise)


# =========================================================
# PARTIE 2 - CONSTRUCTION DES N-GRAMMES
# =========================================================

def construire_unigrammes(corpus_tokenise):
    unigrammes = Counter()
    for phrase in corpus_tokenise:
        unigrammes.update(phrase)
    return unigrammes


def construire_bigrammes(corpus_tokenise):
    bigrammes = Counter()
    for phrase in corpus_tokenise:
        for i in range(len(phrase) - 1):
            bigrammes[(phrase[i], phrase[i + 1])] += 1
    return bigrammes


def construire_trigrammes(corpus_tokenise):
    trigrammes = Counter()
    for phrase in corpus_tokenise:
        for i in range(len(phrase) - 2):
            trigrammes[(phrase[i], phrase[i + 1], phrase[i + 2])] += 1
    return trigrammes


# =========================================================
# Construction des structures globales (corpus principal)
# =========================================================

corpus_tokenise = construire_corpus_tokenise(CORPUS_BRUT)
vocabulaire = construire_vocabulaire(corpus_tokenise)
unigrammes = construire_unigrammes(corpus_tokenise)
bigrammes = construire_bigrammes(corpus_tokenise)
trigrammes = construire_trigrammes(corpus_tokenise)
TOTAL_TOKENS = nombre_total_tokens(corpus_tokenise)
V = len(vocabulaire)  # taille du vocabulaire


# =========================================================
# PARTIE 3 - MODELE UNIGRAMME
# =========================================================

def probabilite_unigramme(mot):
    return unigrammes[mot] / TOTAL_TOKENS


# =========================================================
# PARTIE 3 - MODELE BIGRAMME
# =========================================================

def probabilite_bigramme(mot_precedent, mot):
    """P(mot | mot_precedent) = C(mot_precedent, mot) / C(mot_precedent)"""
    c_wp = unigrammes[mot_precedent]
    if c_wp == 0:
        return 0.0
    c_bigramme = bigrammes[(mot_precedent, mot)]
    return c_bigramme / c_wp


# =========================================================
# MODELE TRIGRAMME (pour comparaison Partie 11)
# =========================================================

def probabilite_trigramme(mot1, mot2, mot3):
    """P(mot3 | mot1, mot2) = C(mot1,mot2,mot3) / C(mot1,mot2)"""
    c_bi = bigrammes[(mot1, mot2)]
    if c_bi == 0:
        return 0.0
    c_tri = trigrammes[(mot1, mot2, mot3)]
    return c_tri / c_bi


# =========================================================
# PARTIE 10 - LISSAGE DE LAPLACE
# =========================================================

def probabilite_laplace(mot_precedent, mot):
    """P_Laplace(mot | mot_precedent) = (C(mot_precedent, mot) + 1) / (C(mot_precedent) + V)"""
    c_wp = unigrammes[mot_precedent]
    c_bigramme = bigrammes[(mot_precedent, mot)]
    return (c_bigramme + 1) / (c_wp + V)


# =========================================================
# PARTIE 4 - PREDICTION DU MOT SUIVANT
# =========================================================

def predire_mot_suivant(contexte, laplace=False, top_n=None):
    """contexte : chaîne de caractères, ex: 'le chat'.
    On prend le DERNIER mot du contexte comme mot precedent (modèle bigramme)."""
    mots = contexte.lower().strip().split()
    if not mots:
        return []
    mot_precedent = mots[-1]

    candidats = {}
    for mot in vocabulaire:
        if mot == "<s>":
            continue
        if laplace:
            p = probabilite_laplace(mot_precedent, mot)
        else:
            p = probabilite_bigramme(mot_precedent, mot)
        if p > 0:
            candidats[mot] = p

    candidats_tries = sorted(candidats.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        candidats_tries = candidats_tries[:top_n]
    return candidats_tries


# =========================================================
# PARTIE 5 - GENERATION DE PHRASE
# =========================================================

def generer_phrase(max_len=15):
    phrase = ["<s>"]
    while phrase[-1] != "</s>" and len(phrase) < max_len:
        mot_precedent = phrase[-1]
        candidats = predire_mot_suivant(mot_precedent)
        if not candidats:
            break
        meilleur_mot = candidats[0][0]
        phrase.append(meilleur_mot)
    return " ".join(phrase)


# =========================================================
# PARTIE 6 - PROBABILITE D'UNE PHRASE
# =========================================================

def probabilite_phrase(phrase, laplace=False, verbose=False):
    tokens = tokeniser(phrase)
    p = 1.0
    details = []
    for i in range(len(tokens) - 1):
        if laplace:
            pi = probabilite_laplace(tokens[i], tokens[i + 1])
        else:
            pi = probabilite_bigramme(tokens[i], tokens[i + 1])
        details.append(((tokens[i], tokens[i + 1]), pi))
        p *= pi
    if verbose:
        for (w1, w2), pi in details:
            print(f"  P({w2} | {w1}) = {pi:.4f}")
    return p


# =========================================================
# PARTIE 7 - COMPARAISON DE PHRASES
# =========================================================

def comparer_phrases(phrase1, phrase2, laplace=False):
    p1 = probabilite_phrase(phrase1, laplace=laplace)
    p2 = probabilite_phrase(phrase2, laplace=laplace)
    return p1, p2


# =========================================================
# PARTIE 8 - CORRECTION CONTEXTUELLE
# =========================================================

corpus_correction_tok = construire_corpus_tokenise(CORPUS_CORRECTION)
vocab_correction = construire_vocabulaire(corpus_correction_tok)
unigrammes_correction = construire_unigrammes(corpus_correction_tok)
bigrammes_correction = construire_bigrammes(corpus_correction_tok)
V_correction = len(vocab_correction)


def probabilite_laplace_correction(mot_precedent, mot):
    c_wp = unigrammes_correction[mot_precedent]
    c_bi = bigrammes_correction[(mot_precedent, mot)]
    return (c_bi + 1) / (c_wp + V_correction)


def corriger_phrase(phrase, position_index, candidats):
    """
    position_index : index du mot (dans la phrase tokenisée AVEC <s>...</s>) à corriger
    candidats : liste de mots candidats, ex: ['sept', 'cet']
    Compare P(candidat | mot_precedent) * P(mot_suivant | candidat)
    """
    tokens = tokeniser(phrase)
    mot_precedent = tokens[position_index - 1]
    mot_suivant = tokens[position_index + 1] if position_index + 1 < len(tokens) else None

    scores = {}
    for c in candidats:
        p = probabilite_laplace_correction(mot_precedent, c)
        if mot_suivant:
            p *= probabilite_laplace_correction(c, mot_suivant)
        scores[c] = p

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =========================================================
# AFFICHAGE / MENU (PARTIE 12 - MINI PROJET)
# =========================================================

def afficher_vocabulaire():
    print(f"\nVocabulaire ({len(vocabulaire)} mots) :")
    print(sorted(vocabulaire))


def afficher_unigrammes():
    print("\nUnigrammes et fréquences :")
    for mot, freq in unigrammes.most_common():
        print(f"  {mot} : {freq}")


def afficher_bigrammes():
    print("\nBigrammes et fréquences :")
    for bg, freq in bigrammes.most_common():
        print(f"  {bg} : {freq}")


def afficher_trigrammes():
    print("\nTrigrammes et fréquences :")
    for tg, freq in trigrammes.most_common():
        print(f"  {tg} : {freq}")


def menu():
    while True:
        print("\n=========================================")
        print("MINI MODELE DE LANGAGE")
        print("=========================================")
        print("1. Afficher le vocabulaire")
        print("2. Afficher les unigrammes")
        print("3. Afficher les bigrammes")
        print("4. Afficher les trigrammes")
        print("5. Calculer une probabilité (bigramme)")
        print("6. Prédire le mot suivant")
        print("7. Générer une phrase")
        print("8. Calculer la probabilité d'une phrase")
        print("9. Corriger une phrase (ex: sept/cet)")
        print("10. Comparer deux phrases")
        print("11. Quitter")
        print("=========================================")
        choix = input("Choix : ").strip()

        if choix == "1":
            afficher_vocabulaire()
        elif choix == "2":
            afficher_unigrammes()
        elif choix == "3":
            afficher_bigrammes()
        elif choix == "4":
            afficher_trigrammes()
        elif choix == "5":
            w1 = input("Mot précédent : ").strip().lower()
            w2 = input("Mot : ").strip().lower()
            print(f"P({w2} | {w1}) = {probabilite_bigramme(w1, w2):.4f}")
        elif choix == "6":
            ctx = input("Contexte : ").strip()
            candidats = predire_mot_suivant(ctx, top_n=5)
            print(f"Candidats après '{ctx}':")
            for mot, p in candidats:
                print(f"  {mot} : {p:.4f}")
        elif choix == "7":
            print("Phrase générée :", generer_phrase())
        elif choix == "8":
            phrase = input("Phrase : ").strip()
            p = probabilite_phrase(phrase, verbose=True)
            print(f"P(phrase) = {p:.10f}")
        elif choix == "9":
            phrase = input("Phrase (ex: 'il a cet ans') : ").strip()
            pos = int(input("Index du mot à corriger (0=<s>) : "))
            cands = input("Candidats séparés par des virgules : ").split(",")
            cands = [c.strip() for c in cands]
            resultat = corriger_phrase(phrase, pos, cands)
            print("Résultats (du plus probable au moins probable) :")
            for mot, p in resultat:
                print(f"  {mot} : {p:.6f}")
        elif choix == "10":
            p1 = input("Phrase 1 : ").strip()
            p2 = input("Phrase 2 : ").strip()
            r1, r2 = comparer_phrases(p1, p2)
            print(f"P(S1) = {r1:.10f}")
            print(f"P(S2) = {r2:.10f}")
        elif choix == "11":
            print("Au revoir.")
            break
        else:
            print("Choix invalide.")


if __name__ == "__main__":
    menu()
