# YOUR CODE HERE
import sys
import math

def main():
    input = sys.stdin.read().strip()
    X = int(input)
    
    # Calculate the ceiling of X / 10
    if X >= 0:
        # For non-negative X, use integer division and adjust if there's a remainder
        result = X // 10
        if X % 10 != 0:
            result += 1
    else:
        # For negative X, use integer division and adjust if there's a remainder
        result = X // 10
        if X % 10 != 0:
            result -= 1
    
    print(result)

if __name__ == "__main__":
    main()