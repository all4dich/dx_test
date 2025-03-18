from dx_engine import InferenceEngine
import sys
# Get model path from an argument
model_path = sys.argv[1]

if __name__ == "__main__":
    engine = InferenceEngine(model_path)
    print(engine)