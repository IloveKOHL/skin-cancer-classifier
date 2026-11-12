"""User-facing texts of the demo in English and German.

The class explanations are short plain-language descriptions for readers without
medical background. They are general information, not advice for an individual case.
"""

from __future__ import annotations

# class code -> (short name, plain-language explanation)
CLASSES = {
    "en": {
        "benign": (
            "Benign (harmless group)",
            "The model sorts the image into the harmless group: mole, benign keratosis "
            "(e.g. age wart), dermatofibroma or vascular lesion.",
        ),
        "malignant": (
            "Malignant or pre-cancerous",
            "The model sorts the image into the group that needs treatment: melanoma, "
            "basal cell carcinoma or actinic keratosis.",
        ),
        "nv": (
            "Mole (melanocytic nevus) - benign",
            "An ordinary mole: a very common, harmless cluster of pigment cells. Most "
            "people have many of them. A mole that changes should be shown to a doctor.",
        ),
        "bkl": (
            "Benign keratosis (e.g. age wart) - benign",
            "A harmless thickening of the top skin layer, for example an age wart "
            "(seborrheic keratosis) or an age spot. Common in older people.",
        ),
        "df": (
            "Dermatofibroma - benign",
            "A small, firm, harmless nodule in the skin, often on the legs.",
        ),
        "vasc": (
            "Vascular lesion (e.g. angioma) - benign",
            "A spot made of small blood vessels, for example a cherry angioma. "
            "Usually red or purple and usually harmless.",
        ),
        "mel": (
            "Melanoma (black skin cancer) - malignant",
            "A malignant tumour of the pigment cells. It can spread to other organs and "
            "is the most dangerous skin cancer in this list. Found early, it can "
            "usually be treated well.",
        ),
        "bcc": (
            "Basal cell carcinoma (a 'white' skin cancer) - malignant",
            "The most common skin cancer. It grows slowly and very rarely spreads, "
            "but it damages the surrounding skin and needs treatment.",
        ),
        "akiec": (
            "Actinic keratosis / Bowen's disease - pre-cancerous",
            "Skin damage caused by years of sun exposure, often a rough, scaly patch. "
            "It is an early or preliminary stage of a 'white' skin cancer and should "
            "be treated.",
        ),
    },
    "de": {
        "benign": (
            "Gutartig (harmlose Gruppe)",
            "Das Modell ordnet das Bild der harmlosen Gruppe zu: Muttermal, gutartige "
            "Keratose (z. B. Alterswarze), Dermatofibrom oder Gefäßveränderung.",
        ),
        "malignant": (
            "Bösartig oder Krebsvorstufe",
            "Das Modell ordnet das Bild der Gruppe zu, die behandelt werden muss: "
            "Melanom, Basalzellkarzinom oder aktinische Keratose.",
        ),
        "nv": (
            "Muttermal (melanozytärer Nävus) - gutartig",
            "Ein gewöhnliches Muttermal bzw. ein Leberfleck: eine sehr häufige, harmlose "
            "Ansammlung von Pigmentzellen. Die meisten Menschen haben viele davon. "
            "Verändert sich ein Muttermal, sollte es ein Arzt ansehen.",
        ),
        "bkl": (
            "Gutartige Keratose (z. B. Alterswarze) - gutartig",
            "Eine harmlose Verdickung der obersten Hautschicht, zum Beispiel eine "
            "Alterswarze (seborrhoische Keratose) oder ein Altersfleck. Häufig bei "
            "älteren Menschen.",
        ),
        "df": (
            "Dermatofibrom - gutartig",
            "Ein kleiner, fester, harmloser Knoten in der Haut, oft an den Beinen.",
        ),
        "vasc": (
            "Gefäßveränderung (z. B. Blutschwämmchen) - gutartig",
            "Eine Stelle aus kleinen Blutgefäßen, zum Beispiel ein Blutschwämmchen "
            "(Angiom). Meist rot oder violett und in der Regel harmlos.",
        ),
        "mel": (
            "Melanom (schwarzer Hautkrebs) - bösartig",
            "Ein bösartiger Tumor der Pigmentzellen. Er kann in andere Organe streuen "
            "und ist der gefährlichste Hautkrebs in dieser Liste. Früh erkannt ist er "
            "meist gut behandelbar.",
        ),
        "bcc": (
            "Basalzellkarzinom (weißer Hautkrebs) - bösartig",
            "Der häufigste Hautkrebs. Er wächst langsam und streut sehr selten, "
            "schädigt aber die umliegende Haut und muss behandelt werden.",
        ),
        "akiec": (
            "Aktinische Keratose / Morbus Bowen - Krebsvorstufe",
            "Eine Hautschädigung durch jahrelange Sonneneinstrahlung, oft eine raue, "
            "schuppige Stelle. Sie ist eine Vor- oder Frühform von weißem Hautkrebs "
            "und sollte behandelt werden.",
        ),
    },
}

UI = {
    "en": {
        "page_title": "Skin lesion classifier (research demo)",
        "title": "# Skin lesion classifier - research demo",
        "disclaimer": (
            "## ⚠️ Not a medical device - research and education only\n\n"
            "This demo is a student learning project. It **cannot diagnose skin cancer** "
            "and is **not a substitute for a dermatologist**. The model was trained on a "
            "small public dataset of dermatoscopic images (mostly light skin) and will be "
            "wrong on many images - especially ordinary phone photos. "
            "**If a skin lesion worries you, see a doctor.**"
        ),
        "image": "Dermatoscopic image",
        "button": "Classify",
        "model_binary": "Model 'benign or malignant' (2 groups)",
        "model_multiclass": "Model 'exact type' (7 diagnoses)",
        "model_ensemble": "Combined model (average of {n} models)",
        "result": "Model output",
        "meaning": "What this term means",
        "score": "Model score for this result",
        "score_help": (
            "how strongly the model leans towards this result - not the probability "
            "that it is correct"
        ),
        "group": "Combined score of the three malignant / pre-cancerous types",
        "not_diagnosis": (
            "This is the output of an experimental model, **not a diagnosis**. The model "
            "is often wrong and often far too confident. Only a doctor can assess a skin "
            "lesion."
        ),
        "scores": "Scores for all possible results",
        "gradcam": "Heat map (Grad-CAM): red = areas that influenced the result most",
        "glossary": "Glossary: what do the terms mean?",
        "footer": (
            "Scores are uncalibrated model outputs, not clinical risk estimates. "
            "Different models can disagree on the same image."
        ),
    },
    "de": {
        "page_title": "Hautläsions-Klassifikator (Forschungsdemo)",
        "title": "# Hautläsions-Klassifikator - Forschungsdemo",
        "disclaimer": (
            "## ⚠️ Kein Medizinprodukt - nur für Forschung und Ausbildung\n\n"
            "Diese Demo ist ein Lernprojekt eines Auszubildenden. Sie **kann keinen "
            "Hautkrebs diagnostizieren** und **ersetzt keinen Hautarzt**. Das Modell wurde "
            "mit einem kleinen öffentlichen Datensatz aus Dermatoskop-Aufnahmen (fast nur "
            "helle Haut) trainiert und liegt bei vielen Bildern falsch - besonders bei "
            "normalen Handyfotos. **Wenn dir eine Hautstelle Sorgen macht, geh zum Arzt.**"
        ),
        "image": "Dermatoskop-Aufnahme",
        "button": "Auswerten",
        "model_binary": "Modell 'gutartig oder bösartig' (2 Gruppen)",
        "model_multiclass": "Modell 'genaue Art' (7 Diagnosen)",
        "model_ensemble": "Kombiniertes Modell (Durchschnitt aus {n} Modellen)",
        "result": "Ausgabe des Modells",
        "meaning": "Was der Begriff bedeutet",
        "score": "Modellwert für dieses Ergebnis",
        "score_help": (
            "wie stark das Modell zu diesem Ergebnis neigt - nicht die Wahrscheinlichkeit, "
            "dass es stimmt"
        ),
        "group": "Zusammengezählter Wert der drei bösartigen Arten bzw. Vorstufen",
        "not_diagnosis": (
            "Das ist die Ausgabe eines experimentellen Modells, **keine Diagnose**. Das "
            "Modell irrt sich oft und ist dabei häufig viel zu selbstsicher. Eine "
            "Hautstelle kann nur ein Arzt beurteilen."
        ),
        "scores": "Werte für alle möglichen Ergebnisse",
        "gradcam": "Wärmebild (Grad-CAM): rot = Bereiche mit dem größten Einfluss",
        "glossary": "Glossar: Was bedeuten die Begriffe?",
        "footer": (
            "Die Werte sind unkalibrierte Modellausgaben, keine Risikoangaben. "
            "Verschiedene Modelle können dasselbe Bild unterschiedlich einordnen."
        ),
    },
}


def glossary_markdown(language: str, class_names: list[str]) -> str:
    """Bullet list explaining every class the loaded models can output."""
    return "\n".join(
        f"- **{CLASSES[language][name][0]}**: {CLASSES[language][name][1]}"
        for name in class_names
    )
