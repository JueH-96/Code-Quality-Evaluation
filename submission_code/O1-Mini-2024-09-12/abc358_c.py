import sys
from itertools import combinations

def main():
    N, M = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(N)]
    flavor_sets = []
    for s in S:
        bit = 0
        for i, c in enumerate(s):
            if c == 'o':
                bit |= 1 << i
        flavor_sets.append(bit)
    full = (1 << M) - 1
    for k in range(1, N+1):
        for comb in combinations(flavor_sets, k):
            union = 0
            for bit in comb:
                union |= bit
            if union == full:
                print(k)
                return
    print(N)

if __name__ == "__main__":
    main()