"""
StarRupture Material Dependency Graph - Standalone HTML Export
Generates a self-contained HTML file that can be opened in any browser.
Run with: python export_dag.py
"""
import json
from collections import defaultdict
import networkx as nx
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label, CustomJS, TapTool
from bokeh.palettes import Category20
from bokeh.layouts import column

# Import the functions from dag_app
import sys
sys.path.insert(0, '/Users/jeremy9959/GitHub/StarRupture')
from dag_app import (
    _extract_recipes, 
    compute_levels, 
    build_dependency_graph,
    layered_positions,
    create_bokeh_graph
)

JSON_PATH = "recipes_2_simplified.json"


def main():
    """Generate standalone HTML visualization using exact same logic as server."""
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    recipes = _extract_recipes(data)
    if not recipes:
        raise RuntimeError("No recipes detected in JSON (unexpected structure).")

    levels = compute_levels(recipes)
    G_full = build_dependency_graph(recipes)

    resolved = {m: lvl for m, lvl in levels.items() if isinstance(lvl, int)}

    # Force helium to level 2 and recalculate dependent materials
    if "helium" in resolved:
        print(f"Helium found. Current level: {resolved['helium']}")
        resolved["helium"] = 2
        print("Helium set to level 2. Recalculating dependent materials...")
        produced_by = defaultdict(list)
        for r in recipes:
            produced_by[r["output"]].append(r)
        changed = True
        iterations = 0
        max_iterations = 10000
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for material in list(resolved.keys()):
                recipe_list = produced_by.get(material, [])
                if any(len(r["inputs"]) == 0 for r in recipe_list):
                    continue
                if not recipe_list:
                    continue
                candidates = []
                for r in recipe_list:
                    in_levels = []
                    for inp in r["inputs"].keys():
                        in_level = resolved.get(inp, 0)
                        in_levels.append(in_level)
                    if in_levels:
                        candidates.append(1 + max(in_levels))
                if candidates:
                    new_level = min(candidates)
                    if new_level != resolved[material]:
                        resolved[material] = new_level
                        changed = True
        print(f"Recalculation complete after {iterations} iterations")

    keep = {m for m, lvl in resolved.items() if lvl <= 4}

    G = G_full.subgraph(keep).copy()

    nx.set_node_attributes(G, {m: resolved[m] for m in G.nodes()}, "level")

    material_to_machine = {}
    for r in recipes:
        material_to_machine[r["output"]] = r["machine"] or "Gathered"
    nx.set_node_attributes(G, material_to_machine, "machine")

    nodes_by_level = defaultdict(list)
    for n in G.nodes():
        nodes_by_level[G.nodes[n]["level"]].append(n)

    if "helium" in nodes_by_level[0]:
        nodes_by_level[0].remove("helium")
        if 2 not in nodes_by_level:
            nodes_by_level[2] = []
        nodes_by_level[2].append("helium")
        G.nodes["helium"]["level"] = 2

    pos = layered_positions(nodes_by_level, G)

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        rs = d.get("recipes", [])
        if len(rs) == 1:
            # Get output rate from source node and input rate for this edge
            source_output_rate = G.nodes[u].get("output_rate", 1)
            input_q = rs[0]["qty"]
            edge_labels[(u, v)] = f"{source_output_rate:g}→{input_q:g}"
        else:
            edge_labels[(u, v)] = f"{len(rs)} recipes"

    # Print level summary
    print("\nMaterial levels:")
    for lvl in range(0, 5):
        items = sorted([m for m, L in resolved.items() if L == lvl])
        print(f"  Level {lvl}: {len(items)} items")

    # Create the plot
    p = create_bokeh_graph(G, pos, edge_labels)
    
    # Set up output to file
    output_file("StarRupture_DAG.html", title="Star Rupture Materials Dependency Graph")
    
    # Save to HTML
    save(column(p, sizing_mode="stretch_both"))
    print("\nHTML file saved as: StarRupture_DAG.html")


if __name__ == "__main__":
    main()

