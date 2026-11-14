# Ton application Airbnb sur Streamlit

**Tu as déjà une première version sur GitHub ? Lis MISE_A_JOUR.md : conserve assets, les dépendances exportées et tes secrets. Aucun nouvel entraînement n’est requis.**

## 1. Préparer le dossier
Décompresse ce ZIP. Place le dossier `streamlit_airbnb` dans le dossier de ton projet, à côté de ton notebook final. Ne le mets pas dans `hf_space`.

Tu n'as plus besoin d'exécuter l'ancienne section 17, qui concernait Hugging Face.

## 2. Exporter depuis le notebook déjà exécuté
Dans VS Code, conserve le noyau qui contient tes données et modèles. Les cellules jusqu'à la fin de la section 13 doivent avoir été exécutées. Tu n'as pas besoin de relancer les démonstrations, l'évaluation ou Gradio.

Ajoute une cellule et exécute :

```python
from pathlib import Path
exec(Path("streamlit_airbnb/export_streamlit.py").read_text(encoding="utf-8"), globals())
```

Cet export utilise les données en mémoire (`reviews`, `cal`, etc.), même si tes fichiers d'origine portent un autre nom. Il conserve les modèles que TU as sélectionnés et entraînés, sans refaire de recherche de paramètres.

Il crée `streamlit_airbnb/assets`, actualise `requirements.txt` avec les versions de ton environnement et produit `airbnb_streamlit_ready.zip`. Les fichiers binaires sont découpés en morceaux de 20 Mio pour l'envoi par navigateur sur GitHub. Il faut tous les conserver.

Si des variables manquent, relance les cellules nécessaires du notebook dans l'ordre. Si une bibliothèque comme SHAP manque, installe-la dans le noyau du notebook puis exécute à nouveau sa cellule.

## 3. Vérifier localement (conseillé)
Dans le terminal VS Code, depuis le dossier du projet, avec ton environnement actif :

```powershell
python -m pip install -r streamlit_airbnb/requirements.txt
python -m streamlit run streamlit_airbnb/app.py
```

Si `python` n'est toujours pas reconnu, utilise le chemin complet de l'interpréteur sélectionné dans ton notebook, précédé de `&` dans PowerShell.

L'application comporte le formulaire de brief et marque, les graphiques, la stratégie STP, les deux validations du gestionnaire, le journal réel des outils et handoffs, les retours de révision, le téléchargement JSON et le simulateur. L'analyse CLIP des photos est désactivée dans cette version légère. L'analyse NLP conserve les résultats de ton notebook ; aucun modèle NLP lourd n'est téléchargé au démarrage.

## 4. Déposer sur GitHub
Connecte-toi à https://github.com et crée un dépôt (New repository), par exemple `airbnb-bordeaux`.

Décompresse `airbnb_streamlit_ready.zip`. Envoie son contenu à la racine du dépôt : `app.py`, `engine_source.py`, `live_runtime.py`, `studio_views.py`, `studio.css`, `requirements.txt` et le dossier `assets` complet. Dans GitHub : Add file → Upload files, puis Commit changes. Tu peux aussi envoyer les autres fichiers du dossier, sauf des clés et secrets. Ne dépose pas le ZIP lui-même. Conserve `assets` comme dossier ; ne déplace pas ses morceaux à la racine.

Le notebook, les CSV bruts et l'ancien dossier `hf_space` ne sont pas nécessaires pour cette application. Ton export contient toutefois des données et des extraits de commentaires : un dépôt public les rend accessibles. N'inclus pas ta mémoire personnelle `agent_memory.json` ni de clé API.

## 5. Déployer
Va sur https://share.streamlit.io, connecte ton compte GitHub et choisis Create app / Deploy an app.

- Repository : ton-compte/airbnb-bordeaux
- Branch : main
- Main file path : app.py

Dans Advanced settings, sélectionne la même version majeure/mineure de Python que celle affichée lors de l'export. Ton message d'erreur indique Python 3.14. Si cette version n'est pas proposée ou si une dépendance est incompatible, utilise une version disponible et compatible dans VS Code, réexécute ton notebook, puis refais l'export ; ne change pas simplement les versions des modèles sérialisés.

Ajoute dans Secrets, si tu veux activer Gemini :

```toml
GEMINI_API_KEY = "ta-cle"
GEMINI_MODEL = "gemini-2.5-flash"
```

Choisis un modèle disponible sur ton compte Google AI Studio. Sans clé, ou en cas d'échec de Gemini, les agents utilisent les règles de secours prévues par ton notebook. Le compte API et ses quotas sont indépendants de l'hébergement Streamlit.

Clique sur Deploy. Après le démarrage, tu obtiendras ton adresse en `.streamlit.app`. Streamlit Community Cloud propose l'hébergement gratuit, sous ses limites de ressources. Une application trop volumineuse peut nécessiter une réduction des données ; la compatibilité et la mémoire seront à confirmer avec ton export réel.

## Limites et validation
Le code de l'interface est adapté du notebook final fourni. Les données CSV et les modèles exécutés n'étaient pas joints : l'export réel doit être généré sur ton ordinateur, où ils existent. Le bon fonctionnement avec tes données et Gemini doit donc être vérifié après cet export. Aucun déploiement externe n'a été effectué.

Les modèles et données sont partagés en lecture ; le contexte des agents, les validations et la mémoire sont propres à chaque session. Cette mémoire n'est pas persistante. Le calendrier décrit la disponibilité, pas des réservations confirmées. Les résultats et limites statistiques du notebook restent applicables.

Documentation :
- https://docs.streamlit.io/deploy/streamlit-community-cloud
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
