import sys
from collections import Counter

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n, k = map(int, sys.stdin.readline().split())
        s = sys.stdin.readline().strip()
        m = n - k
        required_parity = m % 2
        q = (m - required_parity) // 2
        freq = Counter(s)
        sum_pairs = sum(v // 2 for v in freq.values())
        if sum_pairs >= q:
            print("YES")
        else:
            print("NO")

if __name__ == "__main__":
    main()