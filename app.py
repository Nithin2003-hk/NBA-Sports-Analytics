
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CourtVision | NBA Analytics",
    page_icon="🏀",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "nba_xgboost_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")
DATA_PATH = os.path.join(BASE_DIR, "team_games.csv")
MAPPING_PATH = os.path.join(BASE_DIR, "team_mapping.csv")
RESULTS_PATH = BASE_DIR


# --------------------------------------------------
# 2. LOAD FILES
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    mapping = pd.read_csv(MAPPING_PATH)

    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])
    data["SEASON"] = pd.to_numeric(data["SEASON"])

    return data, mapping

@st.cache_data
def load_results():
    metrics = pd.read_csv(
        os.path.join(RESULTS_PATH, "overall_test_metrics.csv")
    )

    seasons = pd.read_csv(
        os.path.join(RESULTS_PATH, "season_performance.csv")
    )

    confidence = pd.read_csv(
        os.path.join(RESULTS_PATH, "confidence_analysis.csv")
    )

    predictions = pd.read_csv(
        os.path.join(RESULTS_PATH, "test_predictions.csv")
    )

    return metrics, seasons, confidence, predictions


try:
    model = load_model()

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    feature_columns = joblib.load(FEATURE_PATH)

    team_data, team_mapping = load_data()

    metrics, season_performance, confidence, predictions = load_results()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# --------------------------------------------------
# 3. SIDEBAR
# --------------------------------------------------

st.sidebar.title("🏀 CourtVision")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Match Prediction",
        "Team Performance",
        "Model Evaluation"
    ]
)

st.sidebar.divider()

st.sidebar.caption("NBA Historical Analytics")
st.sidebar.caption("Dataset: 2003–2025")
st.sidebar.caption("Model: XGBoost")


# --------------------------------------------------
# 4. OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("🏀 CourtVision")
    st.subheader("NBA Match Analytics & Prediction Platform")

    st.write(
        "An interactive basketball analytics platform "
        "using historical NBA team performance and machine learning."
    )

    st.caption(
        "Historical dataset cutoff: 12 February 2025"
    )

    c1, c2, c3, c4 = st.columns(4)

    accuracy = metrics.loc[
        metrics["Metric"] == "Accuracy", "Value"
    ].iloc[0]

    auc = metrics.loc[
        metrics["Metric"] == "ROC-AUC", "Value"
    ].iloc[0]

    brier = metrics.loc[
        metrics["Metric"] == "Brier Score", "Value"
    ].iloc[0]

    c1.metric("Test Accuracy", f"{accuracy:.2%}")
    c2.metric("ROC-AUC", f"{auc:.4f}")
    c3.metric("Brier Score", f"{brier:.4f}")
    c4.metric("NBA Teams", team_data["TEAM_ID"].nunique())

    st.divider()

    st.subheader("Project Dataset")

    d1, d2, d3 = st.columns(3)

    d1.metric("Team-game records", f"{len(team_data):,}")
    d2.metric("Historical seasons", team_data["SEASON"].nunique())
    d3.metric("Model input features", len(feature_columns))

    st.subheader("Model Performance")

    st.dataframe(metrics, use_container_width=True)

    st.info(
        "The model was trained on historical matches from 2003–2018, "
        "validated on 2019–2021 and tested on 2022–2023. "
        "These are historical evaluation results, not live NBA forecasts."
    )


# --------------------------------------------------
# 5. MATCH PREDICTION
# --------------------------------------------------

elif page == "Match Prediction":

    st.title("🏀 Match Prediction")

    st.write(
        "Estimate the probability of a home-team victory "
        "using historical performance before a selected fixture date."
    )

    teams = sorted(team_mapping["TEAM_NAME"].dropna().unique())

    seasons = sorted(team_data["SEASON"].dropna().unique())

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox(
            "Select Home Team",
            teams,
            index=teams.index("Boston Celtics")
            if "Boston Celtics" in teams else 0
        )

    with col2:
        away_team = st.selectbox(
            "Select Away Team",
            teams,
            index=teams.index("Los Angeles Lakers")
            if "Los Angeles Lakers" in teams else 1
        )

    selected_season = st.selectbox(
        "Select NBA Season",
        seasons,
        index=len(seasons) - 1
    )

    season_dates = team_data.loc[
        team_data["SEASON"] == selected_season,
        "GAME_DATE"
    ]

    min_date = season_dates.min().date()
    max_date = season_dates.max().date()

    fixture_date = st.date_input(
        "Fixture date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    st.caption(
        "The selected date must fall within the historical dataset. "
        "Only completed games strictly before this date are used."
    )

    if home_team == away_team:
        st.warning("Please select two different teams.")

    elif st.button("Generate Prediction", type="primary"):

        home_id = team_mapping.loc[
            team_mapping["TEAM_NAME"] == home_team,
            "TEAM_ID"
        ].iloc[0]

        away_id = team_mapping.loc[
            team_mapping["TEAM_NAME"] == away_team,
            "TEAM_ID"
        ].iloc[0]

        cutoff = pd.Timestamp(fixture_date)

        def get_team_features(team_id):

            history = team_data[
                (team_data["TEAM_ID"] == team_id) &
                (team_data["SEASON"] == selected_season) &
                (team_data["GAME_DATE"] < cutoff)
            ].copy()

            history = history.sort_values("GAME_DATE")

            if len(history) < 5:
                return None

            recent = history.tail(5)

            last_game_date = history["GAME_DATE"].max()

            season_win_pct = history["WIN"].mean()

            rest_days = (cutoff - last_game_date).days

            result = {
                "SEASON_WIN_PCT": season_win_pct,
                "REST_DAYS": rest_days,
                "LAST5_POINTS_FOR": recent["POINTS_FOR"].mean(),
                "LAST5_POINTS_AGAINST": recent["POINTS_AGAINST"].mean(),
                "LAST5_FG_PCT": recent["FG_PCT"].mean(),
                "LAST5_FG3_PCT": recent["FG3_PCT"].mean(),
                "LAST5_FT_PCT": recent["FT_PCT"].mean(),
                "LAST5_ASSISTS": recent["ASSISTS"].mean(),
                "LAST5_REBOUNDS": recent["REBOUNDS"].mean(),
                "LAST5_WIN": recent["WIN"].mean()
            }

            return result

        home = get_team_features(home_id)
        away = get_team_features(away_id)

        if home is None or away is None:

            st.error(
                "At least one team has fewer than five completed games "
                "in the selected season before this date. "
                "Choose a later fixture date."
            )

        else:

            values = {
                "HOME_SEASON_WIN_PCT": home["SEASON_WIN_PCT"],
                "HOME_REST_DAYS": home["REST_DAYS"],
                "HOME_LAST5_POINTS_FOR": home["LAST5_POINTS_FOR"],
                "HOME_LAST5_POINTS_AGAINST": home["LAST5_POINTS_AGAINST"],
                "HOME_LAST5_FG_PCT": home["LAST5_FG_PCT"],
                "HOME_LAST5_FG3_PCT": home["LAST5_FG3_PCT"],
                "HOME_LAST5_FT_PCT": home["LAST5_FT_PCT"],
                "HOME_LAST5_ASSISTS": home["LAST5_ASSISTS"],
                "HOME_LAST5_REBOUNDS": home["LAST5_REBOUNDS"],
                "HOME_LAST5_WIN": home["LAST5_WIN"],

                "AWAY_SEASON_WIN_PCT": away["SEASON_WIN_PCT"],
                "AWAY_REST_DAYS": away["REST_DAYS"],
                "AWAY_LAST5_POINTS_FOR": away["LAST5_POINTS_FOR"],
                "AWAY_LAST5_POINTS_AGAINST": away["LAST5_POINTS_AGAINST"],
                "AWAY_LAST5_FG_PCT": away["LAST5_FG_PCT"],
                "AWAY_LAST5_FG3_PCT": away["LAST5_FG3_PCT"],
                "AWAY_LAST5_FT_PCT": away["LAST5_FT_PCT"],
                "AWAY_LAST5_ASSISTS": away["LAST5_ASSISTS"],
                "AWAY_LAST5_REBOUNDS": away["LAST5_REBOUNDS"],
                "AWAY_LAST5_WIN": away["LAST5_WIN"],

                "DIFF_SEASON_WIN_PCT":
                    home["SEASON_WIN_PCT"] - away["SEASON_WIN_PCT"],

                "DIFF_REST_DAYS":
                    home["REST_DAYS"] - away["REST_DAYS"],

                "DIFF_LAST5_POINTS_FOR":
                    home["LAST5_POINTS_FOR"] - away["LAST5_POINTS_FOR"],

                "DIFF_LAST5_POINTS_AGAINST":
                    home["LAST5_POINTS_AGAINST"] - away["LAST5_POINTS_AGAINST"],

                "DIFF_LAST5_FG_PCT":
                    home["LAST5_FG_PCT"] - away["LAST5_FG_PCT"],

                "DIFF_LAST5_FG3_PCT":
                    home["LAST5_FG3_PCT"] - away["LAST5_FG3_PCT"],

                "DIFF_LAST5_FT_PCT":
                    home["LAST5_FT_PCT"] - away["LAST5_FT_PCT"],

                "DIFF_LAST5_ASSISTS":
                    home["LAST5_ASSISTS"] - away["LAST5_ASSISTS"],

                "DIFF_LAST5_REBOUNDS":
                    home["LAST5_REBOUNDS"] - away["LAST5_REBOUNDS"],

                "DIFF_LAST5_WIN":
                    home["LAST5_WIN"] - away["LAST5_WIN"]
            }

            input_df = pd.DataFrame(
                [[values[col] for col in feature_columns]],
                columns=feature_columns
            )

            probability = model.predict_proba(input_df)[0][1]

            st.divider()

            st.subheader("Prediction Result")

            p1, p2 = st.columns(2)

            p1.metric(home_team, f"{probability:.1%}")
            p2.metric(away_team, f"{1-probability:.1%}")

            if probability >= 0.50:
                st.success(f"Model prediction: {home_team} to win")
            else:
                st.success(f"Model prediction: {away_team} to win")

            st.progress(float(probability))

            st.caption(
                "These are model-estimated probabilities, "
                "not guarantees. Historical calibration and "
                "the limited data cutoff should be considered."
            )


# --------------------------------------------------
# 6. TEAM PERFORMANCE
# --------------------------------------------------

elif page == "Team Performance":

    st.title("📊 Team Performance Explorer")

    teams = sorted(team_mapping["TEAM_NAME"].dropna().unique())

    selected_team = st.selectbox(
        "Choose a team",
        teams
    )

    team_id = team_mapping.loc[
        team_mapping["TEAM_NAME"] == selected_team,
        "TEAM_ID"
    ].iloc[0]

    selected_data = team_data[
        team_data["TEAM_ID"] == team_id
    ].sort_values("GAME_DATE")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Games", len(selected_data))

    c2.metric(
        "Win Percentage",
        f"{selected_data['WIN'].mean():.2%}"
    )

    c3.metric(
        "Average Points",
        f"{selected_data['POINTS'].mean():.1f}"
    )

    st.subheader("Points Scored Over Time")

    chart_data = selected_data.set_index("GAME_DATE")[
        ["POINTS_FOR", "POINTS_AGAINST"]
    ]

    st.line_chart(chart_data)

    st.subheader("Season-wise Team Results")

    season_summary = selected_data.groupby("SEASON").agg(
        Matches=("WIN", "count"),
        Wins=("WIN", "sum"),
        Average_Points=("POINTS_FOR", "mean"),
        Average_Points_Against=("POINTS_AGAINST", "mean"),
        Win_Percentage=("WIN", "mean")
    ).reset_index()

    st.dataframe(
        season_summary.round(3),
        use_container_width=True
    )


# --------------------------------------------------
# 7. MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.title("📈 Model Evaluation")

    st.subheader("Overall Test Metrics")

    st.dataframe(metrics, use_container_width=True)

    st.divider()

    st.subheader("Season-wise Model Performance")

    st.dataframe(
        season_performance,
        use_container_width=True
    )

    st.line_chart(
        season_performance.set_index("Season")[
            ["Accuracy", "Precision", "Recall", "F1_Score"]
        ]
    )

    st.subheader("ROC-AUC by Season")

    st.bar_chart(
        season_performance.set_index("Season")["ROC_AUC"]
    )

    st.divider()

    st.subheader("Prediction Confidence Analysis")

    st.dataframe(
        confidence,
        use_container_width=True
    )

    st.bar_chart(
        confidence.set_index("CONFIDENCE_GROUP")["Accuracy"]
    )

    st.divider()

    st.subheader("Historical Test Predictions")

    st.dataframe(
        predictions.head(100),
        use_container_width=True
    )

    st.caption(
        "Displayed predictions are from the historical test dataset. "
        "They are not live NBA game predictions."
    )
