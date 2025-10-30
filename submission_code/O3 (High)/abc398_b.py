import sys
from collections import Counter

def main() -> None:
    # Read the seven card values
    cards = list(map(int, sys.stdin.read().split()))
    
    # Safety: the problem guarantees 7 numbers, but guard anyway
    if len(cards) != 7:
        # If somehow input is malformed, bail out (not expected in tests)
        print("No")
        return
    
    counts = Counter(cards)
    
    # Look for two distinct values: one with ≥3 cards, another with ≥2 cards
    for x, cx in counts.items():
        if cx >= 3:
            for y, cy in counts.items():
                if y != x and cy >= 2:
                    print("Yes")
                    return
    
    # No suitable pair found
    print("No")

if __name__ == "__main__":
    main()