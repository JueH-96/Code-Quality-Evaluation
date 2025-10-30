# YOUR CODE HERE
import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    for _ in range(Q):
        b, k = map(int, sys.stdin.readline().split())
        distances = sorted([abs(b - a) for a in A])
        print(distances[k - 1])

solve()