"""Data validation, cleaning, feature engineering and preprocessing for hotel booking no-shows."""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

MONTHS = ('January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December')
MONTH_TO_NUM = {month: number for number, month in enumerate(MONTHS, start=1)}


class DataPreparation:
    """
    Class for data validation, cleaning, and feature engineering of the hotel
    booking dataset, following the decisions made in EDA.ipynb.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for data preparation.
    """

    REQUIRED_RAW_COLUMNS = (
        'branch', 'booking_month', 'arrival_month', 'arrival_day', 'checkout_month', 'checkout_day',
        'country', 'first_time', 'room', 'price', 'platform', 'num_adults', 'num_children',
    )

    # Created during cleaning, so they are not expected in the raw CSV.
    ENGINEERED_FEATURES = ('price_sgd', 'stay_nights', 'lead_time_months')

    # EDA §6.1: arrival_month has 88 casing variants of 12 months; the rest are
    # clean today but are standardised the same way to guard against future data.
    CATEGORICAL_COLUMNS = (
        'branch', 'booking_month', 'arrival_month', 'checkout_month', 'country', 'first_time', 'room', 'platform',
    )

    # EDA §6.3: num_adults mixes digits and words.
    ADULTS_MAP = {'1': 1, '2': 2, 'one': 1, 'two': 2}

    # EDA §6.3: whole numbers stored as float only because of the empty row.
    INT_COLUMNS = ('arrival_day', 'checkout_day', 'num_children')

    PRICE_PATTERN = r'^(?P<currency>SGD|USD)\$\s*(?P<amount>[\d.]+)$'

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for data preparation.
        """
        self.config = config

    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes data quality statistics on the raw DataFrame before cleaning,
        so every run is auditable.

        Args:
            df (pd.DataFrame): Raw input DataFrame.
        Returns:
            Dict[str, Any]: Row/column counts, missing-value counts, duplicate
                count, counts of each known data-quality issue, and the target
                class distribution.
        """
        report = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "missing_values_per_column": {
                col: int(count) for col, count in df.isna().sum().items() if count > 0
            },
            "n_duplicate_rows": int(df.duplicated().sum()),
        }
        if 'booking_id' in df.columns:
            report["n_duplicate_booking_ids"] = int(df['booking_id'].dropna().duplicated().sum())
        report["n_non_title_case_values"] = {
            col: int((df[col].dropna() != df[col].dropna().str.strip().str.title()).sum())
            for col in self.CATEGORICAL_COLUMNS if col in df.columns
        }
        if 'num_adults' in df.columns:
            report["n_word_num_adults"] = int(df['num_adults'].astype(str).str.strip().str.lower().isin(['one', 'two']).sum())
        if 'checkout_day' in df.columns:
            report["n_negative_checkout_day"] = int((df['checkout_day'] < 0).sum())
        if 'price' in df.columns:
            report["price_currency_counts"] = {
                str(currency): int(count)
                for currency, count in df['price'].str.extract(self.PRICE_PATTERN)['currency'].value_counts().items()
            }
        target = self.config.get("target_column")
        if target in df.columns:
            report["target_distribution"] = {
                str(label): int(count) for label, count in df[target].value_counts(dropna=False).items()
            }

        logger.info("Data validation report: %s", report)
        return report

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the raw bookings and engineers model features.

        Steps (see EDA.ipynb for the evidence behind each one):
            1. Drop exact duplicate rows.
            2. Drop rows with a missing target (one completely empty row, EDA §4).
            3. Fix data-quality issues: casing, word-form adults, negative
               checkout days, text prices in two currencies (EDA §4.1, §6).
            4. Engineer `stay_nights` and `lead_time_months` (EDA §8).
            5. Drop the identifier (`columns_to_drop`).

        Missing `room` and `price_sgd` values are NOT filled here: they are
        filled inside the model pipeline by `RoomPriceImputer`, so the fill
        statistics are learned from the training split only (EDA §4.2-4.3).

        Args:
            df (pd.DataFrame): Input DataFrame containing raw data.
        Returns:
            pd.DataFrame: The cleaned DataFrame (a new object; the input is not modified).
        Raises:
            ValueError: If the DataFrame is empty or is missing a required raw column.
        """
        if df.empty:
            raise ValueError("Cannot clean an empty DataFrame.")

        required_columns = [self.config['target_column'], *self.REQUIRED_RAW_COLUMNS]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Data is missing required raw column(s): {missing_columns}.")

        logger.info("Starting data cleaning process. Rows before cleaning: %d", len(df))

        n_duplicates = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()
        logger.info("Removed %d duplicate row(s).", n_duplicates)

        target = self.config['target_column']
        n_missing_target = int(df[target].isna().sum())
        df = df.dropna(subset=[target])
        df[target] = df[target].astype(int)
        logger.info("Removed %d row(s) with a missing target.", n_missing_target)

        df = self._fix_data_quality_issues(df)
        df = self._engineer_features(df)

        columns_to_drop = [col for col in self.config['columns_to_drop'] if col in df.columns]
        df = df.drop(columns=columns_to_drop)
        logger.info("Dropped column(s): %s", columns_to_drop)

        df = df.reset_index(drop=True)
        logger.info("Data cleaning process completed. Rows after cleaning: %d", len(df))
        return df

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Builds a new preprocessor for numeric, nominal, and ordinal features.
        Called once per model pipeline so each pipeline owns an independent,
        unfitted preprocessor instance. Columns not listed in the config are
        dropped, so excluded features never reach the model.

        Returns:
            ColumnTransformer: A preprocessor for data transformation.
        """
        numeric_features = self.config['numeric_features']
        nominal_features = self.config['nominal_features']
        ordinal_features = self.config['ordinal_features']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])

        # Rare levels (fewer than min_frequency rows) share one "infrequent"
        # column; unseen levels at inference map to it as well.
        nominal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(
                handle_unknown='infrequent_if_exist',
                min_frequency=self.config['onehot_min_frequency'],
            )),
        ])

        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if nominal_features:
            transformers.append(('cat', nominal_transformer, nominal_features))

        if ordinal_features:
            categories = [self.config['ordinal_categories'][feature] for feature in ordinal_features]
            ordinal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(
                    categories=categories,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                )),
            ])
            transformers.append(('ord', ordinal_transformer, ordinal_features))

        return ColumnTransformer(transformers=transformers, remainder='drop')

    def _fix_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fixes the data-quality issues found in the EDA. All fixes are row-wise
        and need no statistics from the data, so they are safe to apply before
        the train/test split.

        Args:
            df (pd.DataFrame): DataFrame after duplicate / missing-target removal.
        Returns:
            pd.DataFrame: The corrected DataFrame.
        Raises:
            ValueError: If `num_adults` or `price` contains an unrecognised value.
        """
        # EDA §6.2: one spelling per category (e.g. 'APril' -> 'April').
        for col in self.CATEGORICAL_COLUMNS:
            standardised = df[col].str.strip().str.title()
            n_changed = int((df[col].notna() & (df[col] != standardised)).sum())
            df[col] = standardised
            if n_changed:
                logger.info("Standardised casing of %d '%s' value(s).", n_changed, col)

        # EDA §6.3: 'one' / 'two' -> 1 / 2.
        num_adults = df['num_adults'].astype(str).str.strip().str.lower().map(self.ADULTS_MAP)
        unmapped = df.loc[num_adults.isna(), 'num_adults'].unique()
        if len(unmapped):
            raise ValueError(f"Unrecognised num_adults value(s): {list(unmapped)}.")
        logger.info("Converted %d word-form num_adults value(s).",
                    int(df['num_adults'].astype(str).str.strip().str.lower().isin(['one', 'two']).sum()))
        df['num_adults'] = num_adults.astype(int)

        # EDA §6.3: negative checkout days are a sign error (arrive 3 Oct, checkout -4 Oct).
        n_negative = int((df['checkout_day'] < 0).sum())
        df['checkout_day'] = df['checkout_day'].abs()
        logger.info("Took the absolute value of %d negative checkout_day value(s).", n_negative)

        df[list(self.INT_COLUMNS)] = df[list(self.INT_COLUMNS)].astype(int)

        # EDA §4.1: prices are text in SGD or USD; convert everything to SGD.
        # Missing prices stay NaN and are filled by RoomPriceImputer.
        parts = df['price'].str.extract(self.PRICE_PATTERN)
        unparsed = df['price'].notna() & parts['amount'].isna()
        if unparsed.any():
            raise ValueError(f"Unrecognised price value(s): {list(df.loc[unparsed, 'price'].unique()[:5])}.")
        amount = parts['amount'].astype(float)
        df['price_sgd'] = amount.where(parts['currency'] == 'SGD', amount * self.config['usd_to_sgd'])
        logger.info("Converted %d USD price(s) to SGD; %d price(s) missing.",
                    int((parts['currency'] == 'USD').sum()), int(df['price_sgd'].isna().sum()))
        df = df.drop(columns='price')

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the engineered features from EDA §8. Both use only values from
        the same row, so they are safe to compute before the train/test split.

        - `stay_nights`: checkout date minus arrival date. The data has no year,
          so 2000 (a leap year, for 29 February arrivals) is assumed; a checkout
          month before the arrival month means the stay crosses the new year.
        - `lead_time_months`: (arrival month - booking month) mod 12.

        Args:
            df (pd.DataFrame): DataFrame after data-quality fixes.
        Returns:
            pd.DataFrame: The DataFrame with engineered features added.
        Raises:
            ValueError: If a date cannot be built or a stay is not at least one night.
        """
        arrival_month = df['arrival_month'].map(MONTH_TO_NUM)
        checkout_month = df['checkout_month'].map(MONTH_TO_NUM)
        booking_month = df['booking_month'].map(MONTH_TO_NUM)

        crosses_new_year = checkout_month < arrival_month
        arrival_date = pd.to_datetime(
            pd.DataFrame({'year': 2000, 'month': arrival_month, 'day': df['arrival_day']}), errors='coerce',
        )
        checkout_date = pd.to_datetime(
            pd.DataFrame({'year': 2000 + crosses_new_year.astype(int), 'month': checkout_month,
                          'day': df['checkout_day']}),
            errors='coerce',
        )
        n_invalid = int((arrival_date.isna() | checkout_date.isna()).sum())
        if n_invalid:
            raise ValueError(f"{n_invalid} row(s) have an invalid arrival or checkout date.")

        df['stay_nights'] = (checkout_date - arrival_date).dt.days
        n_bad_stays = int((df['stay_nights'] < 1).sum())
        if n_bad_stays:
            raise ValueError(f"{n_bad_stays} row(s) have a stay shorter than one night.")
        logger.info("Engineered stay_nights; %d stay(s) cross the new year.", int(crosses_new_year.sum()))

        df['lead_time_months'] = (arrival_month - booking_month) % 12
        return df


class RoomPriceImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing `price_sgd` and `room` values using statistics learned from
    the training data only (EDA §4.2-4.3). Used as the first step of every
    model pipeline, so cross-validation, validation and test rows never
    influence the fill values.

    - Missing price: the median price of training bookings with the same room
      and branch (falling back to the branch median, then the overall median).
    - Missing room: the room whose price band contains the price, within its
      branch. Cut points sit halfway between neighbouring room bands, learned
      from training rows with a known room and a real price. If the price is
      missing too, the branch's most common room is used.

    Attributes (set by `fit`):
        room_branch_median_ (pd.Series): Median price per (branch, room).
        branch_median_ (pd.Series): Median price per branch.
        overall_median_ (float): Overall median price.
        cut_points_ (Dict[str, Dict[str, list]]): Per branch, rooms ordered by
            price and the price edges between them.
        branch_mode_room_ (pd.Series): Most common room per branch.
        overall_mode_room_ (str): Overall most common room.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'RoomPriceImputer':
        """
        Learns fill values from the training rows.

        Args:
            X (pd.DataFrame): Training features with `branch`, `room` and `price_sgd`.
            y (Any): Ignored.
        Returns:
            RoomPriceImputer: The fitted imputer.
        """
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        known = X.dropna(subset=['room', 'price_sgd'])
        if known.empty:
            raise ValueError("RoomPriceImputer needs training rows with both room and price_sgd.")

        self.room_branch_median_ = known.groupby(['branch', 'room'])['price_sgd'].median()
        self.branch_median_ = X.groupby('branch')['price_sgd'].median()
        self.overall_median_ = float(X['price_sgd'].median())

        self.cut_points_ = {}
        bands = known.groupby(['branch', 'room'])['price_sgd'].agg(['min', 'max'])
        for branch, branch_bands in bands.groupby(level='branch'):
            branch_bands = branch_bands.droplevel('branch').sort_values('min')
            edges = (branch_bands['max'].values[:-1] + branch_bands['min'].values[1:]) / 2
            self.cut_points_[branch] = {'rooms': list(branch_bands.index), 'edges': list(edges)}

        rooms = X.dropna(subset=['room'])
        self.branch_mode_room_ = rooms.groupby('branch')['room'].agg(lambda s: s.mode().iloc[0])
        self.overall_mode_room_ = rooms['room'].mode().iloc[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing rooms, then missing prices.

        Args:
            X (pd.DataFrame): Features with `branch`, `room` and `price_sgd`.
        Returns:
            pd.DataFrame: A copy of X with `room` and `price_sgd` filled.
        """
        X = X.copy()

        # Rooms first (they need the real price); rows missing both get the branch's usual room.
        missing_room = X['room'].isna()
        if missing_room.any():
            X.loc[missing_room, 'room'] = [
                self._room_from_price(branch, price)
                for branch, price in zip(X.loc[missing_room, 'branch'], X.loc[missing_room, 'price_sgd'])
            ]

        missing_price = X['price_sgd'].isna()
        if missing_price.any():
            X.loc[missing_price, 'price_sgd'] = [
                self._price_from_room(branch, room)
                for branch, room in zip(X.loc[missing_price, 'branch'], X.loc[missing_price, 'room'])
            ]
        return X

    def get_feature_names_out(self, input_features: List[str] = None) -> np.ndarray:
        """Returns the input column names unchanged (the imputer only fills values)."""
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)

    def _room_from_price(self, branch: str, price: float) -> str:
        if pd.isna(price) or branch not in self.cut_points_:
            return self.branch_mode_room_.get(branch, self.overall_mode_room_)
        cuts = self.cut_points_[branch]
        return cuts['rooms'][int(np.searchsorted(cuts['edges'], price, side='right'))]

    def _price_from_room(self, branch: str, room: str) -> float:
        if (branch, room) in self.room_branch_median_.index:
            return float(self.room_branch_median_[(branch, room)])
        return float(self.branch_median_.get(branch, self.overall_median_))
