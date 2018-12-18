import sys
import os
projectdir = os.path.dirname(__file__)
app_path = os.path.join(projectdir, 'approaches')
miss_path = os.path.join(projectdir, 'miss_generator')
perf_path = os.path.join(projectdir, 'performance_eval')
sys.path.insert(0, app_path)
sys.path.insert(0, miss_path)
sys.path.insert(0, perf_path)
import missing_generator
import reconstruct
import performance_eval

def main():
    missing_generator.run()
    reconstruct.run()
    performance_eval.run()

if __name__ == "__main__":
    main()