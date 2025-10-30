import sys
from itertools import combinations

def read_input():
    N = int(sys.stdin.readline().strip())
    D = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
    return N, D

def solve(N, D):
    max_weight = 0
    for i in range(1, N+1):
        for comb in combinations(range(N), i):
            weight = sum(D[comb[j-1]][comb[j]] for j in range(1, i))
            max_weight = max(max_weight, weight)
    return max_weight

def main():
    N, D = read_input()
    print(solve(N, D))

if __name__ == "__main__":
    main()