"""comfy_client.py - uses YOUR ComfyUI on another computer (over Tailscale / the network) to make the picture paint.py and
color_paint.py then paint. Nothing is installed here: only HTTP calls to ComfyUI's own API.

    python comfy_client.py test                                   # is ComfyUI there? (its system info, nothing runs)
    python comfy_client.py run workflow_api.json photo.jpg out.png    # sends the photo, runs the workflow, saves the result
    python comfy_client.py run workflow_api.json - out.png            # a workflow without an input picture (text -> picture)

The workflow: in ComfyUI turn on dev mode (Settings) and use "Save (API Format)". If it has a LoadImage node, the photo is
uploaded and put there. The first SaveImage / PreviewImage output is saved as out.png. A random seed in the workflow is kept as
it is (so a run can be repeated); --seed N sets every "seed" input.
Settings (settings.py): COMFY_URL = "http://100.x.y.z:8188"  (the other computer's Tailscale address and ComfyUI's port),
or  --url http://100.x.y.z:8188  on any command.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import settings as S

URL = str(getattr(S, "COMFY_URL", "http://127.0.0.1:8188")).rstrip("/")


def _get(path, timeout=10):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return r.read()


def _post(path, data, ctype="application/json", timeout=30):
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def test():
    try:
        info = json.loads(_get("/system_stats"))
    except Exception as e:
        raise SystemExit(f"ComfyUI does not answer at {URL}: {e}\n  Is Tailscale on (both computers), ComfyUI started with "
                         f"--listen 0.0.0.0, and COMFY_URL right in settings.py?")
    sysi = info.get("system", {})
    print(f"ComfyUI at {URL}: OK  (ComfyUI {sysi.get('comfyui_version', '?')}, python {str(sysi.get('python_version', '?'))[:6]})")
    for d in info.get("devices", []):
        print(f"   {d.get('name', '?')}: {d.get('vram_total', 0) / 1e9:.1f} GB VRAM, {d.get('vram_free', 0) / 1e9:.1f} GB free")


def upload(path):
    """Uploads a picture into ComfyUI's input folder; returns the name it got."""
    name = os.path.basename(path)
    bnd = "----brush" + uuid.uuid4().hex
    body = (f"--{bnd}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n"
            ).encode() + open(path, "rb").read() + f"\r\n--{bnd}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{bnd}--\r\n".encode()
    r = json.loads(_post("/upload/image", body, f"multipart/form-data; boundary={bnd}", timeout=120))
    return r.get("name", name)


_INFO = {}


def node_info(cls):
    """ComfyUI's description of one node type (its inputs in order), cached."""
    if cls not in _INFO:
        _INFO[cls] = json.loads(_get("/object_info/" + urllib.parse.quote(cls, safe=""), timeout=60)).get(cls)
    return _INFO[cls]


def ui_to_api(ui):
    """A workflow as the ComfyUI page saves it ("nodes" + "links") -> the API form /prompt runs. Reroutes are followed,
    muted nodes dropped, bypassed nodes passed through; widget values are matched to the node's inputs in order."""
    nodes = {n["id"]: n for n in ui["nodes"]}
    links = {l[0]: l for l in ui.get("links", [])}               # id, from node, from slot, to node, to slot, type

    def source(link_id):
        """(node id, slot) a link really comes from: through reroutes and bypassed nodes."""
        _, fn, fs, _, _, typ = links[link_id][:6]
        n = nodes[fn]
        if n["type"] == "Reroute" or n.get("mode") == 4:
            ins = [i for i in n.get("inputs", []) if i.get("link") is not None
                   and (n["type"] == "Reroute" or i.get("type") == typ)]
            if ins:
                return source(ins[0]["link"])
        return fn, fs

    api = {}
    for nid, n in nodes.items():
        if n["type"] in ("Reroute", "Note", "MarkdownNote", "PrimitiveNode") or n.get("mode") in (2, 4):
            continue
        info = node_info(n["type"])
        if info is None:
            raise SystemExit(f"ComfyUI has no node type {n['type']} (a custom node not installed?)")
        inputs = {}
        linked = {}
        for i in n.get("inputs", []):
            if i.get("link") is not None and i["link"] in links:
                src = source(i["link"])
                if src[0] in nodes and nodes[src[0]].get("mode") != 2:
                    linked[i["name"]] = [str(src[0]), src[1]]
        wv = n.get("widgets_values")
        if isinstance(wv, dict):
            inputs.update({k: v for k, v in wv.items() if k not in linked})
        elif isinstance(wv, list):
            k = 0
            for sect in ("required", "optional"):
                for name, spec in info.get("input", {}).get(sect, {}).items():
                    typ = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
                    opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
                    is_widget = isinstance(typ, list) or typ in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
                    if not is_widget or k >= len(wv):
                        continue
                    if name not in linked:
                        inputs[name] = wv[k]
                    k += 1
                    if opts.get("control_after_generate") or (name in ("seed", "noise_seed") and k < len(wv)
                                                              and wv[k] in ("fixed", "increment", "decrement", "randomize")):
                        k += 1                                 # the page's "after generate" box, not an input
        inputs.update(linked)
        api[str(nid)] = {"class_type": n["type"], "inputs": inputs}
    return api


def load_workflow(path):
    """A workflow file in either form -> the API form."""
    wf = json.load(open(path, encoding="utf-8"))
    return ui_to_api(wf) if isinstance(wf, dict) and "nodes" in wf else wf


def run(workflow, photo, out, seed=None, wait_s=900):
    wf = load_workflow(workflow) if isinstance(workflow, str) else workflow
    if photo and photo != "-":
        name = upload(photo)
        n_in = 0
        for node in wf.values():
            if node.get("class_type") == "LoadImage":
                node["inputs"]["image"] = name; n_in += 1
        print(f"photo uploaded as {name} ({n_in} LoadImage node(s) set)")
    if seed is not None:
        for node in wf.values():
            for k in ("seed", "noise_seed"):
                if k in node.get("inputs", {}):
                    node["inputs"][k] = int(seed)
    cid = uuid.uuid4().hex
    pid = json.loads(_post("/prompt", json.dumps({"prompt": wf, "client_id": cid}).encode()))["prompt_id"]
    print(f"running on ComfyUI ({pid[:8]})", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        h = json.loads(_get(f"/history/{pid}"))
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                raise SystemExit(f"\nComfyUI reported an error: {json.dumps(st)[:600]}")
            outs = h[pid].get("outputs", {})
            saved = [k for k, v in outs.items() if any(i.get("type") == "output" for i in v.get("images", []))]
            for node_out in [outs[k] for k in saved] or list(outs.values()):   # a SaveImage first, else a preview
                for img in node_out.get("images", []):
                    q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                                "type": img.get("type", "output")})
                    open(out, "wb").write(_get(f"/view?{q}", timeout=120))
                    print(f"\nsaved {out} ({time.time() - t0:.0f} s); used workflow and seed kept in {out}.json")
                    json.dump({"workflow": wf, "photo": photo, "prompt_id": pid}, open(out + ".json", "w"), indent=1)
                    return out
        print(".", end="", flush=True)
        time.sleep(2)
    raise SystemExit(f"\nno picture after {wait_s} s")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--url" in a:                                  # --url http://100.x.y.z:8188 overrides COMFY_URL
        i = a.index("--url")
        URL = a[i + 1].rstrip("/")
        del a[i:i + 2]
    if not a or a[0] == "test":
        test()
    elif a[0] == "run" and len(a) >= 4:
        sd = int(a[a.index("--seed") + 1]) if "--seed" in a else None
        run(a[1], a[2], a[3], sd)
    else:
        raise SystemExit(__doc__)
