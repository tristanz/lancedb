"""Test for pydantic field aliases in to_pydantic method."""

import lancedb
from lancedb.pydantic import LanceModel
from pydantic import Field, computed_field


class ItemWithAliases(LanceModel):
    name: str = Field(alias="item")
    price: float
    distance: float = Field(alias="_distance")

    @computed_field
    def similarity(self) -> float:
        return 1 - self.distance


def test_to_pydantic_with_aliases(tmp_path):
    """Test that to_pydantic works with field aliases."""
    db = lancedb.connect(tmp_path / "test.db")

    data = [
        {"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
        {"vector": [5.9, 26.5], "item": "bar", "price": 20.0},
    ]

    table = db.create_table("test_table", data=data, mode="overwrite")
    result = table.search([100, 100]).distance_type("cosine").limit(2)

    # This should work without raising ValidationError
    pydantic_results = result.to_pydantic(ItemWithAliases)

    assert len(pydantic_results) == 2
    assert isinstance(pydantic_results[0], ItemWithAliases)
    assert pydantic_results[0].name == "foo"
    assert pydantic_results[0].price == 10.0
    assert pydantic_results[1].name == "bar"
    assert pydantic_results[1].price == 20.0


def test_manual_construction_with_aliases():
    """Test that manual construction with aliases works as expected."""
    # This should work as shown in the issue
    items = [
        ItemWithAliases(item="foo", price=10.0, _distance=0.1),
        ItemWithAliases(item="bar", price=20.0, _distance=0.2),
    ]

    assert len(items) == 2
    assert items[0].name == "foo"
    assert items[0].price == 10.0
    assert items[0].similarity == 0.9
