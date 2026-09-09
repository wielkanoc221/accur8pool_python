"""Spojnosc callbackow aplikacji Dash z layoutem.

Test wychwytuje klase bledu, przez ktora aplikacja nie startowala: masowa zamiana
'data' -> 'data1' podmienila nie tylko nazwy plikow, ale i nazwy property komponentow
(dcc.Store.data, DataTable.data, dcc.Download.data).
"""
import pytest

from accur8pool.scripts import labeling


@pytest.fixture(scope="module")
def layout_components():
    components = {}
    for component in labeling.app.layout._traverse():
        component_id = getattr(component, "id", None)
        if isinstance(component_id, str):
            components[component_id] = set(getattr(component, "_prop_names", []))

    return components


def _callback_dependencies():
    for key, callback in labeling.app.callback_map.items():
        outputs = key.strip(".").split("...") if key.startswith("..") else [key]
        for output in outputs:
            component_id, _, prop = output.rpartition(".")
            if component_id:
                yield component_id, prop

        for dependency in list(callback["inputs"]) + list(callback["state"]):
            yield dependency["id"], dependency["property"]


def test_app_has_callbacks():
    assert len(labeling.app.callback_map) == 7


def test_every_callback_target_exists_in_layout(layout_components):
    unknown = {
        (component_id, prop)
        for component_id, prop in _callback_dependencies()
        if component_id not in layout_components
    }

    assert not unknown


def test_every_callback_property_exists_on_its_component(layout_components):
    invalid = {
        (component_id, prop)
        for component_id, prop in _callback_dependencies()
        if component_id in layout_components and prop not in layout_components[component_id]
    }

    assert not invalid
