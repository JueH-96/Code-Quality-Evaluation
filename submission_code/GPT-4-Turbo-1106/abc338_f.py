import sys
from itertools import permutations

def read_input():
    N, M = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(M)]
    return N, M, edges

def find_min_weight_walk(N, M, edges):
    # Create adjacency matrix with infinities where there is no edge
    adj_matrix = [[float('inf')] * N for _ in range(N)]
    for u, v, w in edges:
        adj_matrix[u-1][v-1] = w

    # Try all permutations of vertices to find the minimum walk
    min_weight = float('inf')
    for perm in permutations(range(N)):
        weight = 0
        for i in range(N-1):
            weight += adj_matrix[perm[i]][perm[i+1]]
            if weight >= min_weight:
                break  # No need to continue if already higher than current min
        else:
            # Check if there is an edge from the last to the first vertex
            weight += adj_matrix[perm[-1]][perm[0]]
            if weight < min_weight:
                min_weight = weight

    return min_weight if min_weight != float('inf') else None

def main():
    N, M, edges = read_input()
    min_weight = find_min_weight_walk(N, M, edges)
    if min_weight is not None:
        print(min_weight)
    else:
        print("No")

if __name__ == "__main__":
    main()