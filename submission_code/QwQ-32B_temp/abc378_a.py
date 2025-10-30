import sys
from collections import Counter

def main():
    a = list(map(int, sys.stdin.readline().split()))
    counts = Counter(a)
    total = 0
    for c in counts.values():
        total += c // 2
    print(total)

if __name__ == "__main__":
    main()