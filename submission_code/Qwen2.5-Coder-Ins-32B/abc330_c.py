# YOUR CODE HERE
import sys
import math

def main():
    D = int(sys.stdin.read().strip())
    min_diff = float('inf')
    
    # We only need to check up to sqrt(D) for x
    for x in range(int(math.sqrt(D)) + 1):
        y_squared = D - x * x
        y = int(math.sqrt(y_squared))
        # Check both floor and ceil of sqrt(y_squared)
        for y_candidate in [y, y + 1]:
            if y_candidate * y_candidate == y_squared:
                min_diff = 0
                break
            diff = abs(x * x + y_candidate * y_candidate - D)
            if diff < min_diff:
                min_diff = diff
        if min_diff == 0:
            break
    
    print(min_diff)

if __name__ == "__main__":
    main()