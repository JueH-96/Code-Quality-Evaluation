import sys
from collections import deque

def solve():
    """
    Solves the problem by finding a valid sequence of leaf removals.
    The total score is maximized by any valid procedure.
    """
    # Set a higher recursion limit for safety on deep trees, though the solution is iterative.
    sys.setrecursionlimit(300000)

    # Use fast I/O
    try:
        input = sys.stdin.readline
    except (IOError, ValueError):
        return

    try:
        N = int(input())
    except (IOError, ValueError):
        # Handle empty input case for local testing
        return

    # Adjacency list to represent the tree
    adj = [[] for _ in range(N + 1)]
    # Array to store the degree of each vertex, dynamically updated
    deg = [0] * (N + 1)

    # Read the N-1 edges to build the tree
    for _ in range(N - 1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # Array to store the 2-coloring of the tree
    color = [-1] * (N + 1)

    # Perform a 2-coloring of the tree using BFS.
    # As the graph is a connected tree, a single BFS from any node suffices.
    if N > 0:
        q_color = deque([1])
        color[1] = 0
        while q_color:
            u = q_color.popleft()
            c = color[u]
            for v_neighbor in adj[u]:
                if color[v_neighbor] == -1:
                    color[v_neighbor] = 1 - c
                    q_color.append(v_neighbor)

    # Create two deques to store the current leaves, separated by color.
    leaves0 = deque()
    leaves1 = deque()
    for i in range(1, N + 1):
        if deg[i] == 1:
            if color[i] == 0:
                leaves0.append(i)
            else:
                leaves1.append(i)

    # List to store the resulting pairs
    ans = []

    # The main loop processes leaves until all N/2 pairs are found.
    # The problem guarantees that a valid sequence of operations always exists.
    while len(ans) < N // 2:
        # Process all currently available pairs of leaves.
        while leaves0 and leaves1:
            u = leaves0.popleft()
            v = leaves1.popleft()

            ans.append((u, v))

            # "Remove" the leaves by effectively setting their degrees to 0.
            deg[u] = 0
            deg[v] = 0

            # Update the neighbors of the removed leaves.
            # A neighbor might become a new leaf if its degree drops to 1.

            # Process neighbor of u
            for n_u in adj[u]:
                if deg[n_u] > 0:  # Find the single active neighbor
                    deg[n_u] -= 1
                    if deg[n_u] == 1:
                        if color[n_u] == 0:
                            leaves0.append(n_u)
                        else:
                            leaves1.append(n_u)
                    break  # A leaf has only one active neighbor

            # Process neighbor of v
            for n_v in adj[v]:
                if deg[n_v] > 0:  # Find the single active neighbor
                    deg[n_v] -= 1
                    if deg[n_v] == 1:
                        if color[n_v] == 0:
                            leaves0.append(n_v)
                        else:
                            leaves1.append(n_v)
                    break  # A leaf has only one active neighbor

    # Print the resulting pairs
    for u, v in ans:
        print(u, v)

if __name__ == "__main__":
    solve()