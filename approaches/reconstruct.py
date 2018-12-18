import json
import os
import importlib.util

def run():
    script_dir = os.path.dirname(__file__)
    filename = 'config.json'
    (projectdir, tail) = os.path.split(script_dir)
    abs_file_path = os.path.join(projectdir, filename)

    with open(abs_file_path) as f:
        data = json.load(f)

    approachApath = data.get("ApproachA")
    approachBpath = data.get("ApproachB")

    spec = importlib.util.spec_from_file_location("appA", approachApath)
    appA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(appA)

    spec = importlib.util.spec_from_file_location("appB", approachBpath)
    appB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(appB)

    appA.run()
    appB.run()

if __name__ == "__main__":
    run()