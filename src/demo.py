# -*- coding: utf-8 -*-
"""Démonstration automatique de toutes les parties du TP (sans menu interactif)."""

from mini_modele_langage import (
    CORPUS_BRUT, corpus_tokenise, vocabulaire, unigrammes, bigrammes,
    trigrammes, TOTAL_TOKENS, V, probabilite_bigramme, probabilite_laplace,
    probabilite_unigramme, probabilite_trigramme, predire_mot_suivant,
    generer_phrase, probabilite_phrase, comparer_phrases, corriger_phrase,
    tokeniser
)

def titre(t):
    print("\n" + "=" * 60)
    print(t)
    print("=" * 60)


# ---------- PARTIE 1 ----------
titre("PARTIE 1 - Prétraitement")
for brut, tok in zip(CORPUS_BRUT, corpus_tokenise):
    print(f"{brut}\n  -> {tok}")
print(f"\nTaille du vocabulaire : {len(vocabulaire)}")
print(f"Vocabulaire : {sorted(vocabulaire)}")
print(f"Nombre total de tokens (avec répétitions) : {TOTAL_TOKENS}")

# ---------- PARTIE 2 ----------
titre("PARTIE 2 - N-grammes")
print("\nBigrammes et fréquences :")
for bg, f in bigrammes.most_common():
    print(f"  {bg} : {f}")
print("\nTrigrammes et fréquences :")
for tg, f in trigrammes.most_common():
    print(f"  {tg} : {f}")
print("\nBigramme le plus fréquent :", bigrammes.most_common(1))
print("Trigramme le plus fréquent :", trigrammes.most_common(1))

# ---------- PARTIE 3 ----------
titre("PARTIE 3 - Modèle bigramme : probabilités")
paires = [("le", "chat"), ("le", "chien"), ("chat", "mange"),
          ("chat", "aime"), ("du", "poisson"), ("la", "viande")]
for w1, w2 in paires:
    print(f"P({w2} | {w1}) = {probabilite_bigramme(w1, w2):.4f}")

# ---------- PARTIE 4 ----------
titre("PARTIE 4 - Prédiction du mot suivant")
for ctx in ["le chat", "le chien", "le", "chat"]:
    print(f"\nContexte : '{ctx}'")
    for mot, p in predire_mot_suivant(ctx, top_n=5):
        print(f"  {mot} : {p:.4f}")

print("\nP(chat | le) =", round(probabilite_bigramme("le", "chat"), 4))
print("P(le | chat) =", round(probabilite_bigramme("chat", "le"), 4))

# ---------- PARTIE 5 ----------
titre("PARTIE 5 - Génération de phrases")
for i in range(5):
    print(f"Phrase générée {i+1} :", generer_phrase())

# ---------- PARTIE 6 ----------
titre("PARTIE 6 - Probabilité d'une phrase")
for phrase in ["le chat mange du poisson",
               "le chien mange de la viande",
               "le chat joue dans le jardin"]:
    print(f"\nPhrase : {phrase}")
    p = probabilite_phrase(phrase, verbose=True)
    print(f"  => P(phrase) = {p:.8f}")

# ---------- PARTIE 7 ----------
titre("PARTIE 7 - Comparaison de phrases (ordre des mots)")
s1 = "le chat mange du poisson"
s2 = "poisson le mange chat du"
p1, p2 = comparer_phrases(s1, s2)
print(f"P(S1='{s1}') = {p1:.8f}")
print(f"P(S2='{s2}') = {p2:.8f}")

# ---------- PARTIE 8 ----------
titre("PARTIE 8 - Correction contextuelle (corpus supplémentaire)")
phrase_test = "il a cet ans"
tokens_test = tokeniser(phrase_test)
print("Tokens :", tokens_test)
# <s> il a cet ans </s>  -> index 3 = "cet"
resultat = corriger_phrase(phrase_test, 3, ["sept", "cet"])
print(f"Test : '{phrase_test}' -> comparaison sept / cet en position 3 :")
for mot, p in resultat:
    print(f"  {mot} : {p:.6f}")

# ---------- PARTIE 9 ----------
titre("PARTIE 9 - Probabilités nulles")
paires_nulles = [("chat", "pain"), ("chien", "poisson"), ("le", "viande")]
for w1, w2 in paires_nulles:
    print(f"C({w1}, {w2}) = {bigrammes[(w1, w2)]} -> P({w2}|{w1}) = {probabilite_bigramme(w1, w2):.4f}")

# ---------- PARTIE 10 ----------
titre("PARTIE 10 - Lissage de Laplace")
w1, w2 = "chat", "pain"
p_normale = probabilite_bigramme(w1, w2)
p_laplace = probabilite_laplace(w1, w2)
print(f"P({w2} | {w1}) sans lissage    = {p_normale:.6f}")
print(f"P({w2} | {w1}) avec Laplace    = {p_laplace:.6f}")
print(f"(C({w1})={unigrammes[w1]}, V={V})")

# ---------- PARTIE 11 ----------
titre("PARTIE 11 - Comparaison unigramme / bigramme / trigramme")
print("Modèle unigramme (ignore le contexte) :")
for mot in ["chat", "chien", "mange", "poisson"]:
    print(f"  P({mot}) = {probabilite_unigramme(mot):.4f}")

print("\nModèle bigramme, contexte='chat' :")
for mot, p in predire_mot_suivant("chat", top_n=5):
    print(f"  P({mot}|chat) = {p:.4f}")

print("\nModèle trigramme, contexte=('le','chat') :")
candidats_tri = {}
for w3 in vocabulaire:
    p = probabilite_trigramme("le", "chat", w3)
    if p > 0:
        candidats_tri[w3] = p
for mot, p in sorted(candidats_tri.items(), key=lambda x: x[1], reverse=True):
    print(f"  P({mot}|le,chat) = {p:.4f}")

print("\nContexte='le chat mange' (trigramme, base = 'chat','mange') :")
candidats_tri2 = {}
for w3 in vocabulaire:
    p = probabilite_trigramme("chat", "mange", w3)
    if p > 0:
        candidats_tri2[w3] = p
for mot, p in sorted(candidats_tri2.items(), key=lambda x: x[1], reverse=True):
    print(f"  P({mot}|chat,mange) = {p:.4f}")
