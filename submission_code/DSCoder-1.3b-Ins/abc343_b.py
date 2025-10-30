import sys

def solve():
    N = int(sys.stdin.readline().strip())
    adjacency_matrix = []
    for _ in range(N):
        adjacency_matrix.append(list(map(int, sys.stdin.readline().strip().split())))

    for i in range(1, N+1):
        connected_vertices = [j+1 for j in range(N) if adjacency_matrix[i-1][j] == 1]
        connected_vertices.sort()
        print(' '.join(map(str, connected_vertices)))

solve()