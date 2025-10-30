import sys
from collections import Counter

def main():
    cards = list(map(int, sys.stdin.read().split()))
    counts = Counter(cards)
    freq = sorted(counts.values(), reverse=True)
    if freq == [3, 1] or freq == [2, 2]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()