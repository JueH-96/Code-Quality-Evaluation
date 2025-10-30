import math
import sys

def find_factorial_n(X):
    # Start with N = 1 and calculate factorial iteratively
    N = 1
    factorial = 1
    
    while True:
        factorial *= N
        if factorial == X:
            return N
        N += 1

if __name__ == "__main__":
    input_value = int(sys.stdin.read().strip())
    result = find_factorial_n(input_value)
    print(result)