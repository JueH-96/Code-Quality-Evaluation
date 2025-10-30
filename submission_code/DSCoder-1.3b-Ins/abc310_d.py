import sys

def solve():
    N, T, M = map(int, sys.stdin.readline().split())
    incompatible_pairs = []
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        incompatible_pairs.append((A, B))
    # YOUR CODE HERE

solve()