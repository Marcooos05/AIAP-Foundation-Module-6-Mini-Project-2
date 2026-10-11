import numpy as np
import pandas as pd
import pytest

from src.data_preparation import DataPreparation, RoomPriceImputer


@pytest.fixture
def raw_df():
    """Hand-built raw rows, each exercising one data-quality issue from the EDA."""
    return pd.DataFrame({
        "booking_id": [1, 2, 3, 4, 5],
        "no_show": [0.0, 1.0, 0.0, 1.0, np.nan],
        "branch": ["Changi", "Orchard", "Changi", "Orchard", np.nan],
        "booking_month": ["November", "August", "March", "January", np.nan],
        "arrival_month": ["June", "NOVember", "February", "December", np.nan],
        "arrival_day": [25.0, 28.0, 29.0, 30.0, np.nan],
        "checkout_month": ["June", "November", "March", "January", np.nan],
        "checkout_day": [27.0, -30.0, 2.0, 2.0, np.nan],
        "country": ["Singapore", "Indonesia", "India", "China", np.nan],
        "first_time": ["Yes", "Yes", "No", "Yes", np.nan],
        "room": ["Single", "King", np.nan, "Queen", np.nan],
        "price": ["SGD$ 492.98", "USD$ 1000.00", "SGD$ 700.00", np.nan, np.nan],
        "platform": ["Website", "Website", "Agent", "Email", np.nan],
        "num_adults": ["1", "two", "one", "2", np.nan],
        "num_children": [0.0, 1.0, 2.0, 0.0, np.nan],
    })


class TestGenerateValidationReport:
    def test_reports_known_data_quality_issues(self, base_config, raw_df):
        report = DataPreparation(base_config).generate_validation_report(raw_df)
        assert report["n_rows"] == 5
        assert report["n_non_title_case_values"]["arrival_month"] == 1
        assert report["n_word_num_adults"] == 2
        assert report["n_negative_checkout_day"] == 1
        assert report["price_currency_counts"] == {"SGD": 2, "USD": 1}

    def test_reports_target_distribution(self, base_config, raw_df):
        report = DataPreparation(base_config).generate_validation_report(raw_df)
        assert report["target_distribution"] == {"0.0": 2, "1.0": 2, "nan": 1}


class TestCleanData:
    def test_drops_empty_row_and_identifier(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        assert len(cleaned) == 4
        assert "booking_id" not in cleaned.columns
        assert cleaned["no_show"].dtype == int

    def test_removes_duplicate_rows(self, base_config, raw_df):
        df_with_dupe = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
        assert len(DataPreparation(base_config).clean_data(df_with_dupe)) == 4

    def test_does_not_modify_input(self, base_config, raw_df):
        original = raw_df.copy()
        DataPreparation(base_config).clean_data(raw_df)
        pd.testing.assert_frame_equal(raw_df, original)

    def test_standardises_casing(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        assert cleaned.loc[1, "arrival_month"] == "November"

    def test_converts_num_adults_words(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        assert cleaned["num_adults"].tolist() == [1, 2, 1, 2]

    def test_unknown_num_adults_raises(self, base_config, raw_df):
        raw_df.loc[0, "num_adults"] = "three"
        with pytest.raises(ValueError, match="num_adults"):
            DataPreparation(base_config).clean_data(raw_df)

    def test_fixes_negative_checkout_day(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        assert cleaned.loc[1, "checkout_day"] == 30

    def test_converts_prices_to_sgd_and_keeps_missing(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        assert cleaned.loc[0, "price_sgd"] == pytest.approx(492.98)
        assert cleaned.loc[1, "price_sgd"] == pytest.approx(1370.0)
        assert np.isnan(cleaned.loc[3, "price_sgd"])
        assert "price" not in cleaned.columns

    def test_unknown_price_format_raises(self, base_config, raw_df):
        raw_df.loc[0, "price"] = "EUR 100"
        with pytest.raises(ValueError, match="price"):
            DataPreparation(base_config).clean_data(raw_df)

    def test_engineers_stay_nights(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        # 25->27 Jun; 28->30 Nov (after sign fix); 29 Feb->2 Mar (leap year); 30 Dec->2 Jan (new year)
        assert cleaned["stay_nights"].tolist() == [2, 2, 2, 3]

    def test_engineers_lead_time(self, base_config, raw_df):
        cleaned = DataPreparation(base_config).clean_data(raw_df)
        # Nov->Jun = 7, Aug->Nov = 3, Mar->Feb = 11 (wraps), Jan->Dec = 11
        assert cleaned["lead_time_months"].tolist() == [7, 3, 11, 11]

    def test_invalid_date_raises(self, base_config, raw_df):
        raw_df.loc[0, "arrival_day"] = 31.0  # 31 June does not exist
        with pytest.raises(ValueError, match="invalid arrival or checkout date"):
            DataPreparation(base_config).clean_data(raw_df)

    def test_empty_dataframe_raises(self, base_config, raw_df):
        with pytest.raises(ValueError, match="empty"):
            DataPreparation(base_config).clean_data(raw_df.iloc[0:0])

    def test_missing_required_column_raises(self, base_config, raw_df):
        with pytest.raises(ValueError, match="missing required raw column"):
            DataPreparation(base_config).clean_data(raw_df.drop(columns=["checkout_day"]))


class TestRoomPriceImputer:
    @pytest.fixture
    def train(self):
        return pd.DataFrame({
            "branch": ["Changi"] * 6 + ["Orchard"] * 2,
            "room": ["Single", "Single", "Queen", "Queen", "King", "King", "Single", "King"],
            "price_sgd": [450.0, 550.0, 650.0, 750.0, 850.0, 950.0, 700.0, 1300.0],
        })

    def test_fills_price_with_room_branch_median(self, train):
        imputer = RoomPriceImputer().fit(train)
        test = pd.DataFrame({"branch": ["Changi"], "room": ["Queen"], "price_sgd": [np.nan]})
        assert imputer.transform(test).loc[0, "price_sgd"] == pytest.approx(700.0)

    def test_fills_room_from_price_band(self, train):
        imputer = RoomPriceImputer().fit(train)
        test = pd.DataFrame({
            "branch": ["Changi", "Changi", "Changi", "Orchard"],
            "room": [np.nan] * 4,
            "price_sgd": [500.0, 720.0, 990.0, 1250.0],
        })
        assert imputer.transform(test)["room"].tolist() == ["Single", "Queen", "King", "King"]

    def test_learns_from_training_data_only(self, train):
        imputer = RoomPriceImputer().fit(train)
        test = pd.DataFrame({"branch": ["Changi", "Changi"], "room": ["Queen", "Queen"],
                             "price_sgd": [np.nan, 5000.0]})
        # The 5000 in the test data must not change the learned median (700).
        assert imputer.transform(test).loc[0, "price_sgd"] == pytest.approx(700.0)

    def test_row_missing_both_gets_branch_mode_room(self, train):
        imputer = RoomPriceImputer().fit(train)
        test = pd.DataFrame({"branch": ["Changi"], "room": [np.nan], "price_sgd": [np.nan]})
        filled = imputer.transform(test)
        assert filled.loc[0, "room"] in {"Single", "Queen", "King"}
        assert not filled.isna().any().any()

    def test_leaves_no_missing_values_on_cleaned_data(self, base_config, synthetic_raw_df):
        cleaned = DataPreparation(base_config).clean_data(synthetic_raw_df)
        assert cleaned["room"].isna().any() and cleaned["price_sgd"].isna().any()
        filled = RoomPriceImputer().fit_transform(cleaned)
        assert not filled[["room", "price_sgd"]].isna().any().any()

    def test_does_not_modify_input(self, train):
        test = pd.DataFrame({"branch": ["Changi"], "room": [np.nan], "price_sgd": [500.0]})
        original = test.copy()
        RoomPriceImputer().fit(train).transform(test)
        pd.testing.assert_frame_equal(test, original)


class TestCreatePreprocessor:
    def test_returns_independent_instances(self, base_config):
        data_prep = DataPreparation(base_config)
        assert data_prep.create_preprocessor() is not data_prep.create_preprocessor()

    def test_excludes_unconfigured_columns(self, base_config, synthetic_raw_df):
        cleaned = DataPreparation(base_config).clean_data(synthetic_raw_df)
        filled = RoomPriceImputer().fit_transform(cleaned)
        preprocessor = DataPreparation(base_config).create_preprocessor().fit(filled)
        names = preprocessor.get_feature_names_out()
        for dropped in ("platform", "checkout_month", "booking_month", "arrival_day", "checkout_day"):
            assert not any(dropped in name for name in names)
