# YOUR CODE HERE

import sys

def read_input():
    N = int(sys.stdin.readline().strip())
    A = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
    return N, A

def solve(N, A):
    for i in range(N):
        connected_vertices = [j+1 for j in range(N) if A[i][j] == 1]
        connected_vertices.sort()
        print(' '.join(map(str, connected_vertices)))

N, A = read_input()
solve(N, A)