"""Fallback test runner for environments without pytest (same tests, minimal fixtures)."""
import importlib.util, inspect, os, sys, tempfile, traceback, pathlib

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
ok = fail = 0
for f in sorted(pathlib.Path(root, "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            kw = {"tmp_path": pathlib.Path(tempfile.mkdtemp())} if "tmp_path" in inspect.signature(fn).parameters else {}
            fn(**kw); ok += 1; print("PASS", f.stem, name)
        except Exception:
            fail += 1; print("FAIL", f.stem, name); traceback.print_exc()
print(f"{ok} passed, {fail} failed"); sys.exit(1 if fail else 0)
