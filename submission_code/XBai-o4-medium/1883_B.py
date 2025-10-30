import sys
from collections import Counter

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        m = n - k
        cnt = Counter(s)
        sum_pairs = 0
        for v in cnt.values():
            sum_pairs += v // 2
        max_possible = 2 * sum_pairs + 1
        print("YES" if m <= max_possible else "NO")

if __name__ == "__main__":
    main()