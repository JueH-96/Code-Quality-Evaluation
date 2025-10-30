import math
import sys

def main():
    X = int(sys.stdin.read().strip())
    result = math.ceil(X / 10)
    print(result)

if __name__ == "__main__":
    main()