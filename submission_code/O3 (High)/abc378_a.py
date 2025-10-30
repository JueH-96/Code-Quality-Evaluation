import sys
from collections import Counter

def main() -> None:
    # Read the four integers
    balls = list(map(int, sys.stdin.read().strip().split()))
    
    # Count occurrences of each color
    freq = Counter(balls)
    
    # Each pair consists of two balls of the same color
    # The number of pairs we can form for a color is count // 2
    max_operations = sum(count // 2 for count in freq.values())
    
    # Output result
    print(max_operations)

if __name__ == "__main__":
    main()