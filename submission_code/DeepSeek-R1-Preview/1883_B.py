import sys
from collections import Counter

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        m = n - k
        counts = Counter(s)
        O = sum(1 for c in counts.values() if c % 2 != 0)
        target_odds = m % 2
        delta = target_odds - O
        condition1 = ((O + k) % 2) == target_odds
        condition2 = abs(delta) <= k
        if condition1 and condition2:
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()