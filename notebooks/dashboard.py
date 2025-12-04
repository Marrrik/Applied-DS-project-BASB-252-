import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training Random Forest model to predict the probability of customer's churn
    """)
    return


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    import plotly.express as px
    import plotly.graph_objects as go
    return RandomForestClassifier, cross_val_score, go, mo, pd, px


@app.cell
def _(pd):
    df = pd.read_csv(r'C:\Users\Admin\Desktop\HSE\VSHB\Ml_Torreto\Applied-DS-project-BASB-252-\data\raw\Bank\Bank Customer Churn Prediction.csv').drop(columns=['customer_id'])
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The initial data
    """)
    return


@app.cell
def _(df, mo):
    mo.plain(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Here you can choose which features will be used for training the model and see how your choice effects model quality
    """)
    return


@app.cell
def _(df, mo):
    all_features = [col for col in df.columns if col != "churn"]

    ui_feature_selector = mo.ui.multiselect(
        options=all_features,
        value=all_features,
        label="Select features to use for training:"
    )

    ui_feature_selector

    return (ui_feature_selector,)


@app.cell
def _(df, pd, ui_feature_selector):
    df_clean = df.copy()
    selected_features = ui_feature_selector.value

    if "gender" in selected_features:
        df_clean["gender"] = df_clean["gender"].map({"Female": 0, "Male": 1})
    if "country" in selected_features:
        df_clean = pd.get_dummies(df_clean, columns=["country"])
        selected_features.remove('country')
        selected_features.extend(["country_France", "country_Germany", "country_Spain"])
    return df_clean, selected_features


@app.cell
def _(df_clean, selected_features):
    X, y = df_clean[selected_features], df_clean.churn
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## On this step you can choose model's hyperparameters and also see how your choice effects model quality
    """)
    return


@app.cell
def _(mo):
    ui_rf_trees = mo.ui.number(start=10, stop=300, value=100, label="n_estimators")
    ui_rf_depth = mo.ui.number(start=2, stop=50, value=6, label="max_depth")
    ui_rf_min_samples = mo.ui.number(start=2, stop=20, value=2, label="min_samples_split")

    ui_rf_criterion = mo.ui.dropdown(
        options=["gini", "entropy", "log_loss"],
        label="criterion"
    )

    ui_rf_max_features = mo.ui.dropdown(
        options=["sqrt", "log2", "None"],
        label="max_features"
    )

    mo.vstack([
        mo.md("""
    ### Random Forest Parameters
    - **n_estimators** — number of decision trees (min 10, max 300)  
    - **max_depth** — maximal depth of the tree (min 2, max 50)  
    - **min_samples_split** — minimal number of samples for split (min 2, max 20) 
    - **criterion** — function for split quality estimation 
    - **max_features** — the number of features to consider when looking for the best split 
    """),
        ui_rf_trees,
        ui_rf_depth,
        ui_rf_min_samples,
        ui_rf_criterion,
        ui_rf_max_features
    ])

    return (
        ui_rf_criterion,
        ui_rf_depth,
        ui_rf_max_features,
        ui_rf_min_samples,
        ui_rf_trees,
    )


@app.cell
def _(
    RandomForestClassifier,
    X,
    cross_val_score,
    ui_rf_criterion,
    ui_rf_depth,
    ui_rf_max_features,
    ui_rf_min_samples,
    ui_rf_trees,
    y,
):
    model = RandomForestClassifier(
        n_estimators=ui_rf_trees.value,
        max_depth=ui_rf_depth.value,
        min_samples_split=ui_rf_min_samples.value,
        criterion=ui_rf_criterion.value or "gini",
        max_features=None if ui_rf_max_features.value in ["None", None] else ui_rf_max_features.value,
        random_state=42
    )

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    model_roc_auc = cv_scores.mean()
    return model, model_roc_auc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Here is the final quality score of trained model - Mean cross validation ROC-AUC score
    """)
    return


@app.cell
def _(go, model_roc_auc):
    fig_roc_auc = go.Figure()
    fig_roc_auc.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=float(model_roc_auc),
            gauge={"axis": {"range": [0, 1]}},
            title={"text": "CV ROC-AUC"}
        )
    )
    fig_roc_auc

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Below you can see the ranking of features from most important to less important
    """)
    return


@app.cell
def _(X, model, pd, px, y):
    model.fit(X, y)

    importance_values = model.feature_importances_
    importance_title = "Feature Importances of Random Forest Model"

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance_values
    }).sort_values("importance", ascending=True)

    fig_features = px.bar(
        importance_df.head(15),
        x="importance",
        y="feature",
        orientation="h",
        title=importance_title
    )

    fig_features

    return


@app.cell
def _(mo, model_roc_auc):
    mo.md(f"""
    ### Summary

    **Selected model:** {"Random Forest"}  
    **Cross-validated roc_auc:** `{model_roc_auc:.3f}`  

    """)

    return


if __name__ == "__main__":
    app.run()
