"""Fake store data. In a real system these would be calls to your OMS, PIM and CRM."""
from __future__ import annotations

import os
from datetime import date

TODAY = date.fromisoformat(os.getenv("DEMO_TODAY", "2026-10-04"))

CATALOG: list[dict] = [
    {"sku": "EL-100", "name": "Aria Wireless Earbuds", "category": "electronics", "price": 79.00,
     "in_stock": True, "final_sale": False,
     "tags": ["audio", "wireless", "bluetooth", "earbuds", "running", "workout", "sweatproof"],
     "description": "Bluetooth earbuds with 8 hour battery, sweat resistant, secure fit for running."},
    {"sku": "EL-110", "name": "Aria Pro Noise-Cancelling Headphones", "category": "electronics",
     "price": 249.00, "in_stock": True, "final_sale": False,
     "tags": ["audio", "headphones", "noise", "cancelling", "travel", "flights", "wireless"],
     "description": "Over-ear wireless headphones with active noise cancelling and 30 hour battery."},
    {"sku": "EL-205", "name": "Volt 65W USB-C Charger", "category": "electronics", "price": 39.00,
     "in_stock": True, "final_sale": False,
     "tags": ["charger", "usb-c", "laptop", "phone", "travel"],
     "description": "Compact 65W charger for laptops and phones."},
    {"sku": "EL-300", "name": "Lumen Smart Desk Lamp", "category": "electronics", "price": 59.00,
     "in_stock": False, "final_sale": False,
     "tags": ["lamp", "desk", "light", "office", "smart"],
     "description": "Dimmable desk lamp with app control and warm/cool light."},
    {"sku": "HM-410", "name": "Brewline Pour-Over Coffee Kit", "category": "home", "price": 45.00,
     "in_stock": True, "final_sale": False,
     "tags": ["coffee", "kitchen", "gift", "pour-over", "brewing"],
     "description": "Glass dripper, filters, and a hand grinder. Popular gift for coffee lovers."},
    {"sku": "HM-420", "name": "Kettle Pro Gooseneck Kettle", "category": "home", "price": 69.00,
     "in_stock": True, "final_sale": False,
     "tags": ["kettle", "gooseneck", "coffee", "tea", "kitchen", "gift"],
     "description": "Electric gooseneck kettle with temperature hold, made for pour-over coffee and tea."},
    {"sku": "AP-510", "name": "Trailrunner Waterproof Rain Jacket", "category": "apparel", "price": 139.00,
     "in_stock": True, "final_sale": False,
     "tags": ["jacket", "rain", "waterproof", "hiking", "running", "outdoor"],
     "description": "Light waterproof shell with taped seams and a packable hood."},
    {"sku": "AP-520", "name": "Merino Everyday Crew Sweater", "category": "apparel", "price": 89.00,
     "in_stock": True, "final_sale": False,
     "tags": ["sweater", "wool", "merino", "winter", "gift"],
     "description": "Soft merino crew neck, machine washable."},
    {"sku": "AP-530", "name": "Stride Running Shoes", "category": "apparel", "price": 119.00,
     "in_stock": True, "final_sale": False,
     "tags": ["shoes", "running", "sneakers", "workout"],
     "description": "Cushioned daily trainer for road running."},
    {"sku": "OD-610", "name": "Summit 2-Person Tent", "category": "outdoor", "price": 199.00,
     "in_stock": True, "final_sale": False,
     "tags": ["tent", "camping", "hiking", "outdoor"],
     "description": "Three-season two person tent with aluminium poles and a full rainfly."},
    {"sku": "CL-700", "name": "Clearance Yoga Mat", "category": "fitness", "price": 19.00,
     "in_stock": True, "final_sale": True,
     "tags": ["yoga", "mat", "workout", "fitness"],
     "description": "6mm non-slip mat. Clearance item."},
]
CATALOG_BY_SKU = {p["sku"]: p for p in CATALOG}

ORDERS: dict[str, dict] = {
    "SS-10421": {"customer_id": "C-1001", "status": "shipped", "carrier": "UPS",
                 "tracking": "1Z999AA10123456784", "placed": "2026-09-28", "eta": "2026-10-07",
                 "delivered": None, "items": [{"sku": "EL-110", "qty": 1, "price": 249.00}]},
    "SS-10388": {"customer_id": "C-1001", "status": "delivered", "carrier": "USPS",
                 "tracking": "9400111899223344556677", "placed": "2026-09-10", "eta": "2026-09-14",
                 "delivered": "2026-09-14",
                 "items": [{"sku": "EL-100", "qty": 1, "price": 79.00},
                           {"sku": "AP-520", "qty": 1, "price": 89.00}]},
    "SS-10102": {"customer_id": "C-1002", "status": "delivered", "carrier": "FedEx",
                 "tracking": "612999AA10", "placed": "2026-08-01", "eta": "2026-08-05",
                 "delivered": "2026-08-05", "items": [{"sku": "AP-510", "qty": 1, "price": 139.00}]},
    "SS-10455": {"customer_id": "C-1002", "status": "processing", "carrier": None, "tracking": None,
                 "placed": "2026-10-03", "eta": "2026-10-10", "delivered": None,
                 "items": [{"sku": "HM-410", "qty": 1, "price": 45.00}]},
    "SS-10200": {"customer_id": "C-1003", "status": "delivered", "carrier": "UPS",
                 "tracking": "1Z999AA10999999999", "placed": "2026-09-15", "eta": "2026-09-20",
                 "delivered": "2026-09-20",
                 "items": [{"sku": "OD-610", "qty": 1, "price": 199.00},
                           {"sku": "CL-700", "qty": 1, "price": 19.00}]},
}

CUSTOMERS = {
    "C-1001": {"name": "Priya", "tier": "gold"},
    "C-1002": {"name": "Marcus", "tier": "standard"},
    "C-1003": {"name": "Elena", "tier": "silver"},
}

RETURN_POLICY = {
    "standard_window_days": 30,
    "opened_electronics_window_days": 15,
    "defect_window_days": 90,  # defective / damaged / wrong item
    "final_sale_returnable": False,
    "text": (
        "Most items can be returned within 30 days of delivery. Opened electronics: 15 days. "
        "Defective, damaged-in-shipping or wrong items: 90 days and return shipping is free. "
        "Final-sale items cannot be returned unless defective."
    ),
}

COMPLAINT_CREDIT_BY_SEVERITY = {0: 0, 1: 10, 2: 25, 3: 50}  # USD store credit


def orders_for(customer_id: str) -> dict[str, dict]:
    return {oid: o for oid, o in ORDERS.items() if o["customer_id"] == customer_id}
