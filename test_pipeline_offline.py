"""No-network tests for make_trajectories.py's fragile seams: the API
response shapes that only appear in production (null content, missing
fields) and the diff-fence extraction. Run: python test_pipeline_offline.py"""
import io
import json
import types
import urllib.request

def fake_urlopen_factory(payload):
    def fake_urlopen(req, timeout=None):
        body = json.dumps(payload).encode()
        return io.BytesIO(body)
    return fake_urlopen

def load_module_without_running(path):
    src = open(path).read()
    src = src.split("\nTASKS = json.load")[0]   # defs only, not the run
    mod = types.ModuleType("mt")
    mod.extract_diff = None
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod

import os
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
mt = load_module_without_running("make_trajectories.py")

# 1. null content (reasoning budget spent) -> empty string, no crash
urllib.request.urlopen = fake_urlopen_factory(
    {"model": "served/model", "choices": [{"message": {"content": None}}]})
text, served = mt.teacher("prompt")
assert text == "" and served == "served/model", (text, served)
assert mt.extract_diff(text) is None            # empty reply -> no diff

# 2. content missing entirely -> same
urllib.request.urlopen = fake_urlopen_factory(
    {"choices": [{"message": {}}]})
text, served = mt.teacher("prompt")
assert text == "" and served == mt.TEACHER_MODEL

# 3. normal reply -> diff extracted between fences
fence = chr(96) * 3
reply = "Here you go\n%sdiff\n--- a/x\n+++ b/x\n%s\n" % (fence, fence)
urllib.request.urlopen = fake_urlopen_factory(
    {"model": "m", "choices": [{"message": {"content": reply}}]})
text, _ = mt.teacher("prompt")
assert mt.extract_diff(text) == "--- a/x\n+++ b/x"

# 4. the request always carries a max_tokens budget
captured = {}
def capturing_urlopen(req, timeout=None):
    captured.update(json.loads(req.data.decode()))
    return io.BytesIO(json.dumps(
        {"choices": [{"message": {"content": "x"}}]}).encode())
urllib.request.urlopen = capturing_urlopen
mt.teacher("prompt")
assert captured.get("max_tokens", 0) >= 8192, captured

print("4 offline seam tests pass")
