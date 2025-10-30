import sys
from collections import Counter

def main():
    colors = list(map(int, sys.stdin.read().split()))
    counts = Counter(colors)
    total_pairs = 0
    for count in counts.values():
        total_pairs += count // 2
    print(total_pairs)

if __name__ == "__main__":
    main()