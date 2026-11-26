#!/usr/bin/env python3
"""Seed spendly.db with dummy transactions that mirror the SQLAlchemy model:

    class Transaction(Base):
        __tablename__ = "transactions"
        id: Mapped[uuid.UUID]            -> CHAR(32) PRIMARY KEY   (uuid4().hex)
        source: Mapped[str]              -> VARCHAR(20) NOT NULL    {"navi","phonepe","googlepay"}
        date: Mapped[date]               -> DATE (ISO text)         last ~90 days
        description: Mapped[str]         -> TEXT NOT NULL
        merchant: Mapped[str]            -> VARCHAR(200) NOT NULL
        amount: Mapped[Decimal]          -> NUMERIC(12,2) NOT NULL  always positive, INR
        category: Mapped[str]            -> VARCHAR(50) NOT NULL
        account: Mapped[str]            -> VARCHAR(200) NOT NULL
        note / raw_text: nullable        -> TEXT
        fingerprint: String(64) unique   -> VARCHAR(64) + partial unique index
        created_at / updated_at          -> DATETIME

Note on the model's partial unique index: `postgresql_where=` is a Postgres-only
dialect flag. SQLite translates it to a real partial index (supported since
3.8):  CREATE UNIQUE INDEX ... ON transactions(fingerprint) WHERE fingerprint IS NOT NULL.
If you ever build this table via SQLAlchemy's Base.metadata.create_all() on
sqlite, that WHERE clause is silently dropped and the index becomes fully
unique -- harmless here, because NULLs count as distinct in sqlite.

Run:  python3 src/make_demo_db.py
"""

from __future__ import annotations

import hashlib
import os
import random
import sqlite3
import uuid
from datetime import date, timedelta

from constants import PROJECT_ROOT

random.seed(7)
DB = os.path.join(PROJECT_ROOT, "spendly.db")

CATEGORIES = [
    "Food & Dining",
    "Transport",
    "Groceries",
    "Shopping",
    "Entertainment",
    "Health",
    "Utilities",
    "Education",
    "Finance",
    "Uncategorized",
]
SOURCES = {"phonepe": 45, "googlepay": 35, "navi": 20}  # weights
ACCOUNTS = {
    "phonepe": "PhonePe • Union Bank •• 4821",
    "googlepay": "Google Pay • HDFC Bank •• 7713",
    "navi": "Navi Credit Line •• 3391",
}

# (merchant, category, min_amount, max_amount, weight) -- INR, always positive
MERCHANTS = [
    ("Swiggy", "Food & Dining", 149, 899, 3),
    ("Zomato", "Food & Dining", 129, 799, 3),
    ("Starbucks", "Food & Dining", 220, 650, 1),
    ("Domino's Pizza", "Food & Dining", 299, 749, 1),
    ("Cafe Coffee Day", "Food & Dining", 110, 320, 1),
    ("Uber", "Transport", 89, 449, 3),
    ("Ola", "Transport", 79, 399, 2),
    ("Rapido", "Transport", 29, 119, 2),
    ("Indian Oil Petrol Pump", "Transport", 500, 2400, 1),
    ("Namma Metro", "Transport", 10, 60, 1),
    ("BigBasket", "Groceries", 799, 3499, 2),
    ("Blinkit", "Groceries", 199, 1199, 3),
    ("Zepto", "Groceries", 149, 899, 3),
    ("DMart", "Groceries", 599, 2799, 1),
    ("Amazon", "Shopping", 299, 5999, 2),
    ("Flipkart", "Shopping", 499, 6999, 1),
    ("Myntra", "Shopping", 699, 3499, 1),
    ("Croma", "Shopping", 999, 14999, 1),
    ("Decathlon", "Shopping", 599, 3999, 1),
    ("Netflix", "Entertainment", 649, 649, 1),
    ("Spotify", "Entertainment", 119, 119, 1),
    ("BookMyShow", "Entertainment", 249, 1199, 2),
    ("PVR Cinemas", "Entertainment", 300, 900, 1),
    ("Apollo Pharmacy", "Health", 99, 1499, 2),
    ("Practo", "Health", 299, 999, 1),
    ("1mg", "Health", 149, 899, 1),
    ("Jio Fiber", "Utilities", 999, 999, 1),
    ("Airtel Postpaid", "Utilities", 599, 599, 1),
    ("BESCOM Electricity", "Utilities", 849, 2199, 1),
    ("Bangalore Water Supply", "Utilities", 149, 449, 1),
    ("Udemy", "Education", 449, 799, 1),
    ("Coursera", "Education", 999, 3999, 1),
    ("Kindle Books", "Education", 99, 499, 1),
    ("Groww SIP", "Finance", 3000, 3000, 1),
    ("HDFC Credit Card Bill", "Finance", 3999, 8999, 1),
    ("CRED", "Finance", 500, 2000, 1),
    ("UPI Payment to Ramesh", "Uncategorized", 100, 1500, 1),
    ("Payment to Kiran Kumar", "Uncategorized", 200, 2000, 1),
    ("Transfer to Suresh", "Uncategorized", 500, 3000, 1),
]
_WEIGHTED = [row for row in MERCHANTS for _ in range(row[4])]

# Recurring payments pin to a day of the month: (day, merchant, low, high)
RECURRING = [
    (5, "Netflix", 649, 649),
    (12, "Spotify", 119, 119),
    (15, "Jio Fiber", 999, 999),
    (20, "Airtel Postpaid", 599, 599),
    (28, "BESCOM Electricity", 849, 2199),
    (1, "Groww SIP", 3000, 3000),
    (1, "HDFC Credit Card Bill", 3999, 8999),
]

SCHEMA = """
CREATE TABLE transactions (
    id          CHAR(32)     PRIMARY KEY,          -- uuid4().hex, as SQLAlchemy's Uuid stores it
    source      VARCHAR(20)  NOT NULL,
    date        DATE         NOT NULL,
    description TEXT         NOT NULL,
    merchant    VARCHAR(200) NOT NULL,
    amount      NUMERIC(12,2) NOT NULL CHECK (amount > 0),
    category    VARCHAR(50)  NOT NULL,
    account     VARCHAR(200) NOT NULL,
    note        TEXT,
    raw_text    TEXT,
    fingerprint VARCHAR(64),
    created_at  DATETIME     NOT NULL,
    updated_at  DATETIME     NOT NULL
);
CREATE INDEX ix_transactions_date      ON transactions (date);
CREATE INDEX ix_transactions_category  ON transactions (category);
CREATE INDEX ix_transactions_merchant  ON transactions (merchant);
CREATE INDEX ix_transactions_source    ON transactions (source);
CREATE UNIQUE INDEX uq_transactions_fingerprint
    ON transactions (fingerprint) WHERE fingerprint IS NOT NULL;
"""

DESCRIPTION_TEMPLATES = {
    "phonepe": ["Paid to {m}", "Merchant payment to {m}", "{m} - UPI/PhonePe"],
    "googlepay": ["Paid {m} using Google Pay UPI", "{m} - GPay UPI payment"],
    "navi": ["Navi Card purchase at {m}", "{m} - Navi Card txn"],
}

TODAY = date(2026, 11, 24)
START = TODAY - timedelta(days=90)


def pick_source() -> str:
    return random.choices(list(SOURCES), weights=list(SOURCES.values()))[0]


def make_row(
    txn_date: date, merchant: str, category: str, amount: float, seq: int
) -> tuple:
    source = pick_source()
    description = random.choice(DESCRIPTION_TEMPLATES[source]).format(m=merchant)
    account = ACCOUNTS[source]
    note = random.choice(
        [
            "team dinner",
            "monthly refill",
            "gift",
            "office commute",
            "family",
            "recharge for mom",
            "weekend plans",
            None,
            None,
            None,
            None,
        ]
    )
    dmy = txn_date.strftime("%d-%m-%y")
    raw_text = None
    if random.random() < 0.55:  # simulated bank/UPI SMS
        raw_text = (
            f"Rs.{amount:.2f} debited from {account.split('•')[-1].strip()} "
            f"on {dmy} towards {merchant}. Not you? Call 18001234"
        )
    fingerprint = hashlib.sha256(
        f"{source}|{txn_date}|{merchant}|{amount:.2f}|{seq}".encode()
    ).hexdigest()
    ts = f"{txn_date.isoformat()} {random.randint(6, 23):02d}:{random.randint(0, 59):02d}:00"
    return (
        uuid.uuid4().hex,
        source,
        txn_date.isoformat(),
        description,
        merchant,
        round(amount, 2),
        category,
        account,
        note,
        raw_text,
        fingerprint,
        ts,
        ts,
    )


def build_rows() -> list:
    rows, seq = [], 0
    for offset in range((TODAY - START).days + 1):
        day = START + timedelta(days=offset)
        # recurring payments (skip if the day-of-month hasn't happened yet this month)
        if day.day <= 28:
            for d, merchant, low, high in RECURRING:
                if day.day == d:
                    merchant, category = next(
                        (m[0], m[1]) for m in MERCHANTS if m[0] == merchant
                    )
                    rows.append(
                        make_row(
                            day, merchant, category, random.uniform(low, high), seq
                        )
                    )
                    seq += 1
        # everyday spending: 2-5 transactions per day
        for _ in range(random.randint(2, 5)):
            merchant, category, low, high, _w = random.choice(_WEIGHTED)
            rows.append(
                make_row(day, merchant, category, random.uniform(low, high), seq)
            )
            seq += 1
    return rows


def main() -> None:
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    rows = build_rows()
    con.executemany(
        "INSERT INTO transactions (id, source, date, description, merchant, amount, "
        "category, account, note, raw_text, fingerprint, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    cur = con.cursor()
    summary = {
        "rows": cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0],
        "total_spend_inr": cur.execute(
            "SELECT ROUND(SUM(amount), 2) FROM transactions"
        ).fetchone()[0],
        "by_source": dict(
            cur.execute(
                "SELECT source, COUNT(*) FROM transactions GROUP BY source ORDER BY 2 DESC"
            ).fetchall()
        ),
        "date_range": cur.execute(
            "SELECT MIN(date), MAX(date) FROM transactions"
        ).fetchone(),
        "categories": dict(
            cur.execute(
                "SELECT category, ROUND(SUM(amount), 2) FROM transactions GROUP BY category ORDER BY 2 DESC"
            ).fetchall()
        ),
        "negative_amounts": cur.execute(
            "SELECT COUNT(*) FROM transactions WHERE amount <= 0"
        ).fetchone()[0],
        "dupe_fingerprints": cur.execute(
            "SELECT COUNT(*) FROM (SELECT fingerprint FROM transactions "
            "WHERE fingerprint IS NOT NULL GROUP BY fingerprint HAVING COUNT(*) > 1)"
        ).fetchone()[0],
    }
    con.close()
    print(f"seeded {DB}")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
