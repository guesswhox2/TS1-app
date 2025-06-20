from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Dict, List
from collections import defaultdict
import re
# Initialize Flask app and Faker
app = Flask(__name__)
fake = Faker('fr_FR')
random.seed(42)
np.random.seed(42)

# Constants
TEAMS = {
    "football": ["PSG", "OM", "OL", "FC Barcelona", "Real Madrid"],
    "basketball": ["ASVEL", "Paris Basket", "Lakers", "Warriors"],
    "tennis": ["Roland Garros", "Wimbledon"],
    "MMA": ["UFC", "Bellator", "ONE Championship"]
}
TOXIC_WORDS = {
    'fr': [
        'idiot', 'stupide', 'débile', 'connard', 'salope', 'pute', 'nul',
        'merde', 'enculé', 'fils de pute', 'pd', 'ta gueule', 'mort',
        'crève', 'raté', 'loser', 'haine', 'dégage', 'suce', 'batard'
    ],
    'en': [
        'idiot', 'stupid', 'retard', 'asshole', 'bitch', 'whore', 'suck',
        'shit', 'fuck', 'motherfucker', 'fag', 'shut up', 'die', 'kill',
        'loser', 'hate', 'leave', 'suck', 'bastard'
    ]
}

TOXIC_PATTERNS = [
    r'\b(je\s+hais)\b',
    r'\b(tu\s+es\s+nul)\b',
    r'\b(va\s+crever)\b',
    r'\b(pauvre\s+type)\b',
    r'\b(tu\s+merites\s+de\s+mourir)\b'
]
SPORTS = list(TEAMS.keys())
VIDEO_TYPES = ["highlight", "interview", "training", "analysis", "live"]
SPORT_POPULARITY = {
    "football": 2.0,
    "MMA": 1.8,
    "basketball": 1.5,
    "tennis": 1.3
}

class DataGenerator:
    @staticmethod
    def generate_users(n=100):
        users = []
        for _ in range(n):
            fav_sports = random.sample(SPORTS, k=random.randint(1, 3))
            user_teams = [random.choice(TEAMS[sp]) for sp in fav_sports if sp in TEAMS]

            users.append({
                "user_id": fake.unique.uuid4(),
                "name": fake.name(),
                "email": fake.email(),
                "fav_sports": fav_sports,
                "followed_teams": user_teams,
                "signup_date": fake.date_between(start_date='-1y', end_date='today'),
                "country": fake.country_code()
            })
        return pd.DataFrame(users)

    @staticmethod
    def generate_videos(n=300):
        videos = []
        for i in range(n):
            sport = random.choice(SPORTS)
            video_time = fake.date_time_between(start_date='-3m', end_date='now')

            title = (f"Combat {fake.last_name()} vs {fake.last_name()} - {random.choice(['KO', 'Soumission', 'Décision'])}"
                    if sport == "MMA" else
                    f"{sport.capitalize()} {random.choice(['moment', 'action', 'résumé'])} {fake.date('%d/%m')}")

            base_views = random.randint(100, 10000)
            days_old = (datetime.now() - video_time).days
            age_factor = max(0.1, 1 - (days_old / 90))
            views = int(base_views * SPORT_POPULARITY[sport] * age_factor * random.uniform(0.8, 1.2))
            like_ratio = random.uniform(0.05, 0.20)
            likes = int(views * like_ratio)

            videos.append({
                "video_id": f"vid_{i+1:04d}",
                "title": title,
                "sport": sport,
                "type": random.choice(VIDEO_TYPES),
                "duration_sec": random.randint(15, 1800),
                "upload_date": video_time.isoformat(),
                "author": fake.user_name(),
                "tags": [sport, f"#{sport}", f"#{random.choice(['top10', 'moment', 'combat' if sport == 'MMA' else 'sport'])}"],
                "views": max(views, 100),
                "likes": max(likes, 5),
                "team_mentioned": random.choice([None, random.choice(TEAMS.get(sport, []))])
            })
        return pd.DataFrame(videos)

    @staticmethod
    def generate_interactions(users_df, videos_df, n=1000):
        interactions = []
        duration_map = videos_df.set_index('video_id')['duration_sec'].to_dict()

        for _ in range(n):
            user_id = random.choice(users_df['user_id'])
            video_id = random.choice(videos_df['video_id'])
            duration = duration_map[video_id]
            watch_time = random.randint(1, duration)
            liked = watch_time > (duration * 0.7)

            interactions.append({
                'user_id': user_id,
                'video_id': video_id,
                'watch_time': watch_time,
                'liked': liked,
                'timestamp': fake.date_time_between(start_date='-3m', end_date='now').isoformat()
            })
        return pd.DataFrame(interactions)

    @staticmethod
    def generate_social_connections(users_df, n=500):
        connections = []
        user_ids = users_df['user_id'].tolist()

        for _ in range(n):
            user1, user2 = random.sample(user_ids, 2)
            user1_data = users_df[users_df['user_id'] == user1].iloc[0]
            user2_data = users_df[users_df['user_id'] == user2].iloc[0]

            common_sports = set(user1_data['fav_sports']).intersection(user2_data['fav_sports'])
            same_country = user1_data['country'] == user2_data['country']

            connection_strength = 0.5
            if common_sports:
                connection_strength += 0.2 * len(common_sports)
            if same_country:
                connection_strength += 0.3

            connections.append({
                'user_id': user1,
                'friend_id': user2,
                'connection_strength': min(connection_strength, 1.0),
                'connection_date': fake.date_time_between(start_date='-1y', end_date='now').isoformat()
            })
        return pd.DataFrame(connections)

class RecommendationEngine:
    def __init__(self, users_df, videos_df, interactions_df, social_df):
        self.users_df = users_df
        self.videos_df = videos_df
        self.interactions_df = interactions_df
        self.social_df = social_df

        # Calculate engagement scores
        videos_duration_map = videos_df.set_index('video_id')['duration_sec'].to_dict()
        self.interactions_df['engagement_score'] = self.interactions_df.apply(
            lambda x: self.calculate_engagement_score(x['watch_time'], videos_duration_map.get(x['video_id'], 1)),
            axis=1
        )

        # Initialize friend recommender
        self.friend_recommender = self.create_friend_recommender()

    @staticmethod
    def calculate_engagement_score(watch_time, video_duration):
        raw_score = watch_time / video_duration
        if raw_score > 0.95: return 1.0
        elif raw_score > 0.7: return 0.8
        elif raw_score > 0.3: return 0.5
        else: return 0.2

    def create_friend_recommender(self):
        tfidf = TfidfVectorizer()
        sports_text = self.users_df['fav_sports'].apply(lambda x: ' '.join(x))
        sports_matrix = tfidf.fit_transform(sports_text)

        social_graph = nx.from_pandas_edgelist(
            self.social_df,
            'user_id',
            'friend_id',
            edge_attr='connection_strength',
            create_using=nx.Graph()
        )
        adjacency_matrix = nx.to_numpy_array(social_graph)

        country_matrix = pd.get_dummies(self.users_df['country'])

        return {
            'sports_similarity': cosine_similarity(sports_matrix),
            'social_similarity': cosine_similarity(adjacency_matrix),
            'country_similarity': cosine_similarity(country_matrix),
            'user_index': {i: uid for i, uid in enumerate(self.users_df['user_id'])}
        }

    def recommend_friends(self, user_id, n=5):
        try:
            user_idx = [i for i, uid in self.friend_recommender['user_index'].items() if uid == user_id][0]
            user_data = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        except IndexError:
            return []

        composite_score = (
                0.4 * self.friend_recommender['sports_similarity'][user_idx] +
                0.3 * self.friend_recommender['social_similarity'][user_idx] +
                0.3 * self.friend_recommender['country_similarity'][user_idx]
        )

        existing_friends = self.social_df[self.social_df['user_id'] == user_id]['friend_id'].tolist()
        recommendations = []

        for i, score in sorted(enumerate(composite_score), key=lambda x: x[1], reverse=True):
            friend_id = self.friend_recommender['user_index'][i]
            if friend_id != user_id and friend_id not in existing_friends:
                friend_data = self.users_df[self.users_df['user_id'] == friend_id].iloc[0]

                # Calculate reasons
                common_sports = set(user_data['fav_sports']).intersection(friend_data['fav_sports'])
                common_teams = set(user_data['followed_teams']).intersection(friend_data['followed_teams'])
                same_country = user_data['country'] == friend_data['country']

                reason_parts = []
                if common_sports:
                    reason_parts.append(f"{len(common_sports)} common sports")
                if same_country:
                    reason_parts.append("same country")

                recommendations.append({
                    "user_id": friend_id,
                    "name": friend_data['name'],
                    "common_sports": list(common_sports),
                    "common_teams": list(common_teams),
                    "connection_score": round(score, 2),
                    "reason": ", ".join(reason_parts) if reason_parts else "Similar interests"
                })

                if len(recommendations) >= n:
                    break
        return recommendations

    def recommend_videos(self, user_id, n=5):
        try:
            # 1. Verify user exists
            user_match = self.users_df[self.users_df["user_id"] == user_id]
            if user_match.empty:
                raise ValueError(f"User {user_id} not found")

            user = user_match.iloc[0]

            # 2. Get user preferences with fallbacks
            user_sports = user.get("fav_sports", [])
            if not user_sports:
                # Fallback to popular videos if no sports preferences
                return self.videos_df.nlargest(n, 'views')[['video_id', 'title', 'sport', 'views', 'likes']].assign(
                    match_reason="Popular video (no preferences)"
                )

            followed_teams = user.get('followed_teams', [])

            # 3. Get watched videos and friend recommendations
            watched_videos = self.interactions_df[
                self.interactions_df['user_id'] == user_id
                ]['video_id'].unique()

            friend_recs = self.recommend_friends(user_id)
            friend_ids = [f[0] for f in friend_recs] if friend_recs else []

            # 4. Scoring function
            def video_score(video):
                score = 0
                if video['sport'] in user_sports:
                    score += 20  # Increased weight for sport match
                    if video['team_mentioned'] in followed_teams:
                        score += 30  # Bonus for team match

                # Friend engagement scoring
                if friend_ids:
                    friend_engagements = self.interactions_df[
                        (self.interactions_df['user_id'].isin(friend_ids)) &
                        (self.interactions_df['video_id'] == video['video_id'])
                        ]['engagement_score']
                    if not friend_engagements.empty:
                        score += friend_engagements.mean() * 25

                # Popularity and recency factors
                score += np.log1p(video['views']) * 4  # Better than log10
                upload_date = pd.to_datetime(video['upload_date'])
                days_old = (pd.Timestamp.now() - upload_date).days
                score += max(0, 10 - (days_old / 7))  # Recency bonus

                return score

            # 5. Get candidate videos
            candidate_videos = self.videos_df[
                (~self.videos_df['video_id'].isin(watched_videos))
            ].copy()

            # If no videos in preferred sports, expand selection
            if len(candidate_videos[candidate_videos['sport'].isin(user_sports)]) < n:
                candidate_videos = self.videos_df.copy()

            # Score and sort videos
            candidate_videos['score'] = candidate_videos.apply(video_score, axis=1)
            recommendations = candidate_videos.sort_values('score', ascending=False).head(n)

            # 6. Generate match reasons
            def get_match_reason(row):
                reasons = []
                if row['sport'] in user_sports:
                    reasons.append(f"Favorite sport: {row['sport']}")
                    if row['team_mentioned'] in followed_teams:
                        reasons.append(f"Followed team: {row['team_mentioned']}")
                if friend_ids and not self.interactions_df[
                    (self.interactions_df['user_id'].isin(friend_ids)) &
                    (self.interactions_df['video_id'] == row['video_id'])
                ].empty:
                    reasons.append("Popular with friends")
                return " | ".join(reasons) if reasons else "General recommendation"

            recommendations['match_reason'] = recommendations.apply(get_match_reason, axis=1)

            return recommendations[['video_id', 'title', 'sport', 'views', 'likes', 'match_reason']]

        except Exception as e:
            # Fallback to popular videos if any error occurs
            print(f"Recommendation error for {user_id}: {str(e)}")
            return self.videos_df.nlargest(n, 'views')[['video_id', 'title', 'sport', 'views', 'likes']].assign(
                match_reason="Popular video (fallback)"
            )


class CommentModerator:
    def __init__(self, users_df):
        self.users_df = users_df
        self.user_reputation = defaultdict(int)
        self.banned_users = set()
        self.toxic_comments = []

        # Initialize with some reputation based on signup date
        for _, user in users_df.iterrows():
            days_since_signup = (datetime.now() - datetime.combine(user['signup_date'], datetime.min.time())).days
            self.user_reputation[user['user_id']] = min(50, days_since_signup * 0.5)

    def generate_comments(self, videos_df, n=500):
        comments = []
        video_ids = videos_df['video_id'].tolist()
        user_ids = self.users_df['user_id'].tolist()

        for _ in range(n):
            video_id = random.choice(video_ids)
            user_id = random.choice(user_ids)
            is_toxic = random.random() < 0.15  # 15% chance of toxic comment

            if is_toxic:
                comment = self._generate_toxic_comment()
            else:
                comment = self._generate_normal_comment(videos_df, video_id)

            comments.append({
                'comment_id': fake.unique.uuid4(),
                'video_id': video_id,
                'user_id': user_id,
                'text': comment,
                'timestamp': fake.date_time_between(start_date='-1m', end_date='now').isoformat(),
                'reported': False,
                'moderated': is_toxic  # Pretend we've already caught toxic comments
            })

        return pd.DataFrame(comments)

    def _generate_normal_comment(self, videos_df, video_id):
        video = videos_df[videos_df['video_id'] == video_id].iloc[0]
        sport = video['sport']
        team = video.get('team_mentioned', '')

        templates = [
            f"Super {sport} comme toujours!",
            f"{team} a vraiment bien joué ce match",
            "Quel match incroyable!",
            "Le joueur était en feu aujourd'hui",
            "J'adore cette chaîne pour le {sport}",
            "Quel but incroyable!",
            "La défense était solide aujourd'hui",
            "Quel talent!",
            "C'était un match passionnant",
            "Je ne peux pas croire ce qui vient de se passer!"
        ]

        return random.choice(templates)

    def _generate_toxic_comment(self):
        toxic_type = random.choice(['word', 'phrase', 'threat'])

        if toxic_type == 'word':
            return f"Vous êtes tous des {random.choice(TOXIC_WORDS['fr'])}!"
        elif toxic_type == 'phrase':
            return random.choice([
                "Tu devrais arrêter le sport, t'es nul",
                "Pourquoi est-ce que tu existes?",
                "L'arbitre est un vrai débile",
                "Ton équipe mérite de perdre",
                "Retourne dans ton pays"
            ])
        else:
            return random.choice([
                "Je vais te trouver et te casser la gueule",
                "Crève dans un accident",
                "J'espère que tu meurs bientôt",
                "Ta famille devrait avoir honte"
            ])

    def detect_toxicity(self, text: str) -> Tuple[bool, Dict[str, bool], List[str]]:
        """Detect toxic content in a comment.

        Returns:
            tuple: (is_toxic, flags, matched_patterns)
        """
        text_lower = text.lower()
        flags = {
            'toxic_word': False,
            'toxic_phrase': False,
            'threat': False,
            'hate_speech': False,
            'spam': False
        }
        matched_patterns = []

        # Check for toxic words
        for lang, words in TOXIC_WORDS.items():
            for word in words:
                if word in text_lower:
                    flags['toxic_word'] = True
                    matched_patterns.append(f"toxic_word:{word}")

        # Check for toxic patterns
        for pattern in TOXIC_PATTERNS:
            if re.search(pattern, text_lower):
                flags['toxic_phrase'] = True
                matched_patterns.append(f"toxic_pattern:{pattern}")

        # Check for threats
        threat_phrases = ['je vais te', 'crève', 'meurs', 'casser la gueule', 'ta famille']
        if any(phrase in text_lower for phrase in threat_phrases):
            flags['threat'] = True
            matched_patterns.append("threat_detected")

        # Check for hate speech
        hate_terms = ['race', 'religion', 'handicap', 'gay', 'lesbienne', 'juif', 'noir', 'arabe']
        if any(term in text_lower for term in hate_terms):
            flags['hate_speech'] = True
            matched_patterns.append("hate_speech")

        # Simple spam detection (repeated phrases)
        if len(re.findall(r'\b(\w+)\b.*\b\1\b', text_lower)) > 2:
            flags['spam'] = True
            matched_patterns.append("spam_detected")

        is_toxic = any(flags.values())
        return is_toxic, flags, matched_patterns

    def moderate_comment(self, comment_data: Dict) -> Dict:
        """Process a comment through moderation system."""
        user_id = comment_data['user_id']
        text = comment_data['text']

        # Skip if user is banned
        if user_id in self.banned_users:
            return {
                'approved': False,
                'reason': 'user_banned',
                'action': 'delete'
            }

        # Check toxicity
        is_toxic, flags, patterns = self.detect_toxicity(text)

        # Determine action
        if is_toxic:
            self.user_reputation[user_id] -= 10
            self.toxic_comments.append({
                'comment_id': comment_data['comment_id'],
                'user_id': user_id,
                'text': text,
                'flags': flags,
                'patterns': patterns,
                'timestamp': datetime.now().isoformat()
            })

            # Ban user if reputation too low
            if self.user_reputation[user_id] < -20:
                self.banned_users.add(user_id)
                action = 'delete_and_ban'
            else:
                action = 'delete'

            return {
                'approved': False,
                'reason': 'toxic_content',
                'flags': flags,
                'action': action,
                'reputation_penalty': -10
            }

        # If clean, increase reputation slightly
        self.user_reputation[user_id] += 1

        return {
            'approved': True,
            'reason': 'clean',
            'action': 'approve',
            'reputation_bonus': 1
        }

    def handle_appeal(self, user_id: str, comment_id: str) -> Dict:
        """Process a user appeal for a moderated comment."""
        # Find the toxic comment
        toxic_comment = next(
            (c for c in self.toxic_comments
             if c['comment_id'] == comment_id and c['user_id'] == user_id),
            None
        )

        if not toxic_comment:
            return {
                'success': False,
                'message': 'Comment not found in moderation records'
            }

        # Simple appeal logic - 50% chance of success
        appeal_success = random.random() < 0.5

        if appeal_success:
            self.user_reputation[user_id] += 5  # Partial reputation restore
            self.toxic_comments.remove(toxic_comment)

            return {
                'success': True,
                'message': 'Appeal approved. Comment restored with partial reputation restoration.',
                'reputation_restored': 5
            }
        else:
            return {
                'success': False,
                'message': 'Appeal denied. Original moderation decision stands.'
            }

# Initialize data and recommendation engine
users_df = DataGenerator.generate_users(100)
videos_df = DataGenerator.generate_videos(300)
interactions_df = DataGenerator.generate_interactions(users_df, videos_df)
social_df = DataGenerator.generate_social_connections(users_df)
recommendation_engine = RecommendationEngine(users_df, videos_df, interactions_df, social_df)
moderator = CommentModerator(users_df)
comments_df = moderator.generate_comments(videos_df)
# API Endpoints
@app.route('/recommendations/<user_id>', methods=['GET'])  # Fixed route syntax
def get_recommendations(user_id):
    try:
        limit = int(request.args.get('limit', 5))
        recommendations = recommendation_engine.recommend_videos(user_id, limit)
        if recommendations.empty:
            return jsonify({"warning": "No recommendations available for this user"}), 200
        return jsonify(recommendations.to_dict(orient='records'))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user_row = users_df[users_df['user_id'] == user_id]
        if user_row.empty:
            return jsonify({"error": f"User {user_id} not found"}), 404
        return jsonify(user_row.iloc[0].to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/users', methods=['GET'])
def list_users():
    try:
        limit = int(request.args.get('limit', 10))
        return jsonify(users_df.head(limit).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/friends/suggestions/<user_id>', methods=['GET'])
def friend_suggestions(user_id):  # Fix typo in function name ('suggestions' not 'suggestions')
    try:
        threshold = float(request.args.get('threshold', 0.5))
        suggestions = recommendation_engine.recommend_friends(user_id)
        filtered = [s for s in suggestions if s['connection_score'] >= threshold]
        return jsonify({"suggestions": filtered})  # Ensure only JSON is returned
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/videos', methods=['GET'])
def list_videos():
    try:
        limit = int(request.args.get('limit', 10))
        return jsonify(videos_df.head(limit).to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/comments', methods=['POST'])
def post_comment():
    try:
        data = request.get_json()
        required_fields = ['video_id', 'user_id', 'text']

        # Validation des champs requis
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Vérification de l'existence de l'utilisateur
        if data['user_id'] not in users_df['user_id'].values:
            return jsonify({"error": "User not found"}), 404

        # Vérification de l'existence de la vidéo
        if data['video_id'] not in videos_df['video_id'].values:
            return jsonify({"error": "Video not found"}), 404

        # Création du commentaire
        comment_data = {
            'comment_id': fake.unique.uuid4(),  # Correction de 'uvid4' à 'uuid4'
            'video_id': data['video_id'],
            'user_id': data['user_id'],
            'text': data['text'],
            'timestamp': datetime.now().isoformat()
        }

        # Modération du commentaire
        moderation_result = moderator.moderate_comment(comment_data)

        # Stockage selon le résultat de la modération
        if moderation_result['approved']:
            global comments_df
            comments_df = pd.concat([comments_df, pd.DataFrame([comment_data])], ignore_index=True)
            return jsonify({
                "message": "Comment posted successfully",
                "comment_id": comment_data['comment_id'],
                "moderation_result": moderation_result
            }), 201
        else:
            return jsonify({
                "error": "Comment rejected by moderation system",
                "moderation_result": moderation_result
            }), 403

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/comments/<video_id>', methods=['GET'])
def get_video_comments(video_id):
    try:
        limit = int(request.args.get('limit', 10))
        # Only show approved comments
        video_comments = comments_df[
            (comments_df['video_id'] == video_id) &
            (~comments_df['user_id'].isin(moderator.banned_users))
            ].sort_values('timestamp', ascending=False).head(limit)

        return jsonify(video_comments.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/moderation/appeal', methods=['POST'])
def submit_appeal():
    try:
        data = request.get_json()
        if 'user_id' not in data or 'comment_id' not in data:
            return jsonify({"error": "Missing user_id or comment_id"}), 400

        result = moderator.handle_appeal(data['user_id'], data['comment_id'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/moderation/stats', methods=['GET'])
def get_moderation_stats():
    try:
        stats = {
            "total_comments": len(comments_df),
            "toxic_comments_detected": len(moderator.toxic_comments),
            "banned_users": len(moderator.banned_users),
            "avg_user_reputation": sum(moderator.user_reputation.values()) / len(moderator.user_reputation),
            "appeal_success_rate": 0.5  # Would track real stats in production
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)