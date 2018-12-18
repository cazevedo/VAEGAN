import sys
sys.path.insert(0, '/approaches')
sys.path.insert(0, '/miss_generator')
sys.path.insert(0, '/performance_eval')
import missing_generator
import reconstruct
import performance_eval

def main():
    missing_generator.run()
    reconstruct.run()
    performance_eval.run()

if __name__ == "__main__":
    main()