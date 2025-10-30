def find_min_edges_to_delete(N, M, edges):
    if M == 0:
        return 0  # No edges to delete if there are none

    # A forest can have at most N - 1 edges
    max_edges_in_forest = N - 1
    
    # The number of edges that need to be deleted to make the graph a forest
    edges_to_delete = M - max_edges_in_forest
    
    # If edges_to_delete is negative, it means we already have a forest or less than N-1 edges
    return max(0, edges_to_delete)

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, M = map(int, data[0].split())
    edges = [tuple(map(int, line.split())) for line in data[1:M+1]]
    
    result = find_min_edges_to_delete(N, M, edges)
    print(result)

if __name__ == "__main__":
    main()