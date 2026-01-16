"""
StarRupture Material Dependency Graph - Bokeh Server Application
Run with: bokeh serve dag_app.py
"""
import json
from collections import defaultdict
import networkx as nx
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, Range1d, LabelSet, CustomJS, PointDrawTool
from bokeh.palettes import Category20
from bokeh.layouts import column


JSON_PATH = "recipes_2_simplified.json"


def _extract_recipes(data):
    """Normalize recipes from the JSON into a list of dicts."""
    recipes = []

    def looks_like_recipe(obj):
        if not isinstance(obj, dict):
            return False
        keys = set(obj.keys())
        has_out = any(k in keys for k in ("output", "out", "result", "produces"))
        has_in = any(k in keys for k in ("inputs", "in", "ingredients", "requires", "cost"))
        return has_out and has_in

    def get_output(obj):
        for k in ("output", "out", "result", "produces"):
            if k in obj:
                out = obj[k]
                if isinstance(out, dict) and "id" in out:
                    return out["id"]
                return out
        return None

    def get_inputs(obj):
        for k in ("inputs", "in", "ingredients", "requires", "cost"):
            if k in obj:
                return obj[k]
        return None

    def get_machine(obj):
        for k in ("machine", "building", "station", "producer"):
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        return None

    def normalize_inputs(inp):
        if inp is None:
            return {}
        if isinstance(inp, dict):
            norm = {}
            for k, v in inp.items():
                try:
                    norm[str(k)] = float(v)
                except Exception:
                    norm[str(k)] = 1.0
            return norm
        if isinstance(inp, list):
            norm = {}
            for el in inp:
                if isinstance(el, dict):
                    item = el.get("id") or el.get("item") or el.get("name") or el.get("material")
                    amt = el.get("amount_per_minute") or el.get("amount") or el.get("qty") or el.get("count") or 1
                    if item is not None:
                        try:
                            norm[str(item)] = float(amt)
                        except Exception:
                            norm[str(item)] = 1.0
                elif isinstance(el, (list, tuple)) and len(el) >= 1:
                    item = el[0]
                    amt = el[1] if len(el) > 1 else 1
                    try:
                        norm[str(item)] = float(amt)
                    except Exception:
                        norm[str(item)] = 1.0
            return norm
        return {}

    def crawl(obj, current_machine=None):
        nonlocal recipes
        if isinstance(obj, dict):
            machine = current_machine
            for k in ("machine", "building", "station", "name", "id"):
                if k in obj and isinstance(obj[k], str):
                    machine = obj[k]
                    break

            if looks_like_recipe(obj):
                out = get_output(obj)
                inp = normalize_inputs(get_inputs(obj))
                m = get_machine(obj) or current_machine
                if out is not None:
                    recipes.append({"machine": m, "output": str(out), "inputs": inp})

            for v in obj.values():
                crawl(v, machine)

        elif isinstance(obj, list):
            for v in obj:
                crawl(v, current_machine)

    crawl(data, None)

    uniq = []
    seen = set()
    for r in recipes:
        key = (r["machine"], r["output"], tuple(sorted(r["inputs"].items())))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def compute_levels(recipes):
    produced_by = defaultdict(list)
    all_materials = set()

    for r in recipes:
        out = r["output"]
        ins = set(r["inputs"].keys())
        produced_by[out].append(r)
        all_materials.add(out)
        all_materials |= ins

    level = {m: None for m in all_materials}

    for m, rs in produced_by.items():
        if any(len(r["inputs"]) == 0 for r in rs):
            level[m] = 0

    changed = True
    for _ in range(10_000):
        if not changed:
            break
        changed = False
        for m, rs in produced_by.items():
            if level[m] == 0:
                continue
            candidates = []
            for r in rs:
                ins = r["inputs"]
                if len(ins) == 0:
                    candidates.append(0)
                    continue
                in_levels = []
                ok = True
                for i in ins.keys():
                    if level[i] is None:
                        ok = False
                        break
                    in_levels.append(level[i])
                if ok:
                    candidates.append(1 + max(in_levels))
            if candidates:
                new_lvl = min(candidates)
                if level[m] is None or new_lvl < level[m]:
                    level[m] = new_lvl
                    changed = True
    return level


def build_dependency_graph(recipes):
    G = nx.DiGraph()
    for r in recipes:
        out = r["output"]
        G.add_node(out)
        for inp, qty in r["inputs"].items():
            G.add_node(inp)
            if G.has_edge(inp, out):
                G.edges[inp, out]["recipes"].append({"machine": r["machine"], "qty": qty})
            else:
                G.add_edge(inp, out, recipes=[{"machine": r["machine"], "qty": qty}])
    return G


def layered_positions(nodes_by_level, G):
    """Deterministic layered layout."""
    pos = {}
    for lvl, nodes in sorted(nodes_by_level.items()):
        if lvl == 0:
            nodes_sorted = sorted(nodes)
            if "helium" in nodes_sorted:
                nodes_sorted.remove("helium")
            if "sulfur" in nodes_sorted:
                nodes_sorted.remove("sulfur")
                nodes_sorted = ["sulfur"] + nodes_sorted
            n = len(nodes_sorted)
            for i, node in enumerate(nodes_sorted):
                x = i - (n - 1) / 2
                y = -lvl
                pos[node] = (x, y)
        elif lvl == 1:
            nodes_sorted = sorted(nodes)
            if "block_calcium" in nodes_sorted:
                nodes_sorted.remove("block_calcium")
            if "basic_building" in nodes_sorted:
                nodes_sorted.remove("basic_building")
            final_order = []
            if "block_calcium" in nodes:
                final_order.append("block_calcium")
            final_order.extend(nodes_sorted)
            if "basic_building" in nodes:
                final_order.append("basic_building")
            n = len(final_order)
            for i, node in enumerate(final_order):
                x = i - (n - 1) / 2
                y = -lvl
                pos[node] = (x, y)
        elif lvl == 2:
            level2_nodes = sorted(nodes)
            if "helium" in level2_nodes:
                level2_nodes.remove("helium")
                level2_nodes = ["helium"] + level2_nodes
            spacing = 1.3
            n = len(level2_nodes)
            for i, node in enumerate(level2_nodes):
                x = (i - (n - 1) / 2) * spacing
                y = -lvl
                pos[node] = (x, y)
        elif lvl == 3:
            level3_nodes = sorted(nodes)
            spacing = 1.5
            if "pressurized_helium" in level3_nodes:
                level3_nodes.remove("pressurized_helium")
            n = len(level3_nodes)
            pressurized_x = -(n * spacing) / 2 - spacing
            pos["pressurized_helium"] = (pressurized_x, -lvl)
            for i, node in enumerate(level3_nodes):
                centered = i * spacing - (n - 1) * spacing / 2
                x = centered + spacing / 2
                y = -lvl
                pos[node] = (x, y)
        else:
            nodes = sorted(nodes)
            n = len(nodes)
            for i, node in enumerate(nodes):
                x = i - (n - 1) / 2
                y = -lvl
                pos[node] = (x, y)
    return pos


def _bokeh_sources(G, pos, edge_labels):
    machines = sorted({G.nodes[n].get("machine", "Unknown") for n in G.nodes()})
    palette = Category20[20] if len(machines) <= 20 else Category20[20] * ((len(machines) // 20) + 1)
    machine_to_color = {m: palette[i % len(palette)] for i, m in enumerate(machines)}

    # Build node data as dict of lists (columnar format)
    node_data = {
        "name": [],
        "x": [],
        "y": [],
        "machine": [],
        "level": [],
        "color": [],
        "label": [],
        "in_degree": [],
        "out_degree": [],
    }
    
    for n in G.nodes():
        x, y = pos[n]
        machine = G.nodes[n].get("machine", "Unknown")
        level = G.nodes[n].get("level", "?")
        # Two-line label: material name on top, machine type below
        label = f"{n}\n{machine}" if machine != "Gathered" else n
        
        node_data["name"].append(n)
        node_data["x"].append(x)
        node_data["y"].append(y)
        node_data["machine"].append(machine)
        node_data["level"].append(level)
        node_data["color"].append(machine_to_color.get(machine, "#808080"))
        node_data["label"].append(label)
        node_data["in_degree"].append(G.in_degree(n))
        node_data["out_degree"].append(G.out_degree(n))

    edge_xs = []
    edge_ys = []
    start_labels = []
    end_labels = []
    edge_qty_labels = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_xs.append([x0, x1])
        edge_ys.append([y0, y1])
        start_labels.append(u)
        end_labels.append(v)
        edge_qty_labels.append(edge_labels.get((u, v), ""))

    node_source = ColumnDataSource(node_data)
    edge_source = ColumnDataSource({
        "xs": edge_xs,
        "ys": edge_ys,
        "start": start_labels,
        "end": end_labels,
        "label": edge_qty_labels,
    })
    return node_source, edge_source, machine_to_color


def create_bokeh_graph(G, pos, edge_labels):
    """Create Bokeh figure for server app."""
    node_source, edge_source, machine_to_color = _bokeh_sources(G, pos, edge_labels)
    
    # Store original y-coordinates for level constraint
    node_source.data['y_original'] = list(node_source.data['y'])

    all_x = [x for coords in node_source.data["x"] for x in ([coords] if not isinstance(coords, list) else coords)]
    all_y = [y for coords in node_source.data["y"] for y in ([coords] if not isinstance(coords, list) else coords)]
    x_min, x_max = min(all_x) - 1.0, max(all_x) + 1.0
    y_min, y_max = min(all_y) - 1.0, max(all_y) + 1.0

    p = figure(
        width=1400,
        height=900,
        title="StarRupture Material Dependency Graph (Bokeh Server, Levels 0–4)",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover,tap",
        active_scroll="wheel_zoom",
        x_range=Range1d(x_min, x_max),
        y_range=Range1d(y_min, y_max),
        match_aspect=True,
    )
    
    # Remove grid and axes
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False

    p.multi_line(
        xs="xs",
        ys="ys",
        source=edge_source,
        line_color="#7a7a7a",
        line_alpha=0.65,
        line_width=2,
    )

    renderer = p.circle(
        x="x",
        y="y",
        size=55,
        source=node_source,
        fill_color="color",
        line_color="white",
        line_width=1.2,
        hover_fill_color="color",
        hover_line_color="#222",
        hover_line_width=2,
        selection_line_color="#222",
        selection_line_width=2,
    )

    labels = LabelSet(
        x="x",
        y="y",
        text="label",
        source=node_source,
        text_font_size="5pt",
        text_align="center",
        text_baseline="middle",
        y_offset=0,
    )
    p.add_layout(labels)
    
    # Add CustomJS callback to constrain dragging to horizontal movement only
    callback = CustomJS(args=dict(node_source=node_source, edge_source=edge_source, pos=pos), code="""
        const data = node_source.data;
        const names = data['name'];
        const xs = data['x'];
        const ys = data['y'];
        const y_orig = data['y_original'];
        
        // Constrain y to original level
        for (let i = 0; i < ys.length; i++) {
            ys[i] = y_orig[i];
        }
        
        // Update edge positions based on new node positions
        const edge_data = edge_source.data;
        const edge_xs = edge_data['xs'];
        const edge_ys = edge_data['ys'];
        const starts = edge_data['start'];
        const ends = edge_data['end'];
        
        // Create name->position lookup
        const node_pos = {};
        for (let i = 0; i < names.length; i++) {
            node_pos[names[i]] = [xs[i], ys[i]];
        }
        
        // Update edge coordinates
        for (let i = 0; i < starts.length; i++) {
            const start_name = starts[i];
            const end_name = ends[i];
            if (node_pos[start_name] && node_pos[end_name]) {
                edge_xs[i] = [node_pos[start_name][0], node_pos[end_name][0]];
                edge_ys[i] = [node_pos[start_name][1], node_pos[end_name][1]];
            }
        }
        
        node_source.change.emit();
        edge_source.change.emit();
    """)
    
    node_source.js_on_change('data', callback)
    
    # Add PointDrawTool to enable dragging nodes
    draw_tool = PointDrawTool(renderers=[renderer], add=False)
    p.add_tools(draw_tool)

    hover = HoverTool(
        tooltips=[
            ("material", "@name"),
            ("machine", "@machine"),
            ("level", "@level"),
            ("in → out", "@in_degree → @out_degree"),
        ],
        renderers=[renderer],
        point_policy="follow_mouse"
    )
    p.add_tools(hover)

    edge_hover = HoverTool(
        tooltips=[
            ("edge", "@start → @end"),
            ("qty/recipes", "@label"),
        ],
        renderers=[],
        line_policy="nearest",
        point_policy="none",
    )
    p.add_tools(edge_hover)

    return p


def main():
    """Main function to build and display the DAG."""
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
                    old_level = resolved[material]
                    if new_level != old_level:
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
            q = rs[0]["qty"]
            edge_labels[(u, v)] = f"{q:g}"
        else:
            edge_labels[(u, v)] = f"{len(rs)} recipes"

    # Print level summary
    print("\nMaterial levels:")
    for lvl in range(0, 5):
        items = sorted([m for m, L in resolved.items() if L == lvl])
        print(f"  Level {lvl}: {len(items)} items")

    return create_bokeh_graph(G, pos, edge_labels)


# Create the plot and add to document
plot = main()
curdoc().add_root(column(plot))
curdoc().title = "StarRupture DAG"
