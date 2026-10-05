from retail_mesh.agents.returns_agent import check_eligibility

EARBUDS = {"category": "electronics", "final_sale": False, "price": 79.0}
JACKET = {"category": "apparel", "final_sale": False, "price": 139.0}
MAT = {"category": "fitness", "final_sale": True, "price": 19.0}


def test_defective_gets_90_days():
    assert check_eligibility(EARBUDS, 60, "defective", 0.9)["eligible"]


def test_opened_electronics_15_days():
    assert not check_eligibility(EARBUDS, 20, "changed_mind", 0.9)["eligible"]
    assert check_eligibility(EARBUDS, 20, "changed_mind", 0.1)["eligible"]


def test_standard_window():
    assert not check_eligibility(JACKET, 60, "size_fit", 0.2)["eligible"]


def test_final_sale_only_if_faulty():
    assert not check_eligibility(MAT, 3, "changed_mind", 0.0)["eligible"]
    assert check_eligibility(MAT, 3, "defective", 0.0)["eligible"]
