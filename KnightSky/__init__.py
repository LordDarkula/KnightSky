import importlib

try:
    importlib.import_module("chess_py")
except ImportError:
    import pip
    pip.main(['install', "chess_py"])
finally:
    globals()["chess_py"] = importlib.import_module("chess_py")