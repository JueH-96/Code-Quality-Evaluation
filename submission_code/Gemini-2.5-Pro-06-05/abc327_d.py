import sys
from collections import deque

def main():
    """
    Reads graph information and determines if it is bipartite.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline

        # Read problem size
        N, M = map(int, input().split())

        # Constraints state M >= 1, so these lines will not be empty.
        A = list(map(int, input().split()))
        B = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle potential empty input on some platforms, though
        # competitive programming judges usually guarantee valid input.
        return

    # Create an adjacency list for the graph. Vertices are 1-indexed.
    adj = [[] for _ in range(N + 1)]
    for i in range(M):
        u, v = A[i], B[i]
        adj[u].append(v)
        adj[v].append(u)

    # `colors` array stores the color of each vertex:
    # -1: uncolored, 0: color 0, 1: color 1
    colors = [-1] * (N + 1)
    is_bipartite = True

    # Iterate through all vertices to handle disconnected components
    for i in range(1, N + 1):
        # If vertex `i` is uncolored, start a new BFS traversal
        if colors[i] == -1:
            q = deque([i])
            colors[i] = 0  # Assign the first color

            while q:
                u = q.popleft()
                
                # Check all neighbors of the current vertex `u`
                for v in adj[u]:
                    if colors[v] == -1:
                        # If neighbor `v` is uncolored, assign the opposite color
                        colors[v] = 1 - colors[u]
                        q.append(v)
                    elif colors[v] == colors[u]:
                        # If an adjacent vertex has the same color, an odd-cycle is detected.
                        # The graph is not bipartite.
                        is_bipartite = False
                        break
                
                if not is_bipartite:
                    break
        
        if not is_bipartite:
            break

    # Output the result
    if is_bipartite:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()