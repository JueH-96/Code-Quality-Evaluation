import sys
from collections import Counter

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        freq = Counter(s)
        original_odd = sum(1 for cnt in freq.values() if cnt % 2 != 0)
        min_deletions = max(0, original_odd - 1)
        parity_condition = (original_odd % 2) == ((n - k) % 2)
        if not parity_condition:
            print("NO")
            continue
        if k >= min_deletions and (k - min_deletions) % 2 == 0:
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()