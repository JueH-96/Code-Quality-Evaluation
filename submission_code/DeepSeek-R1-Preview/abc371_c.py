import sys
from itertools import permutations

def main():
    n = int(sys.stdin.readline())

    mg = int(sys.stdin.readline())
    g_edges = set()
    for _ in range(mg):
        u, v = map(int, sys.stdin.readline().split())
        if u > v:
            u, v = v, u
        g_edges.add((u, v))

    mh = int(sys.stdin.readline())
    h_edges = set()
    for _ in range(mh):
        a, b = map(int, sys.stdin.readline().split())
        if a > b:
            a, b = b, a
        h_edges.add((a, b))

    # Read the A matrix
    A = {}
    current_i = 1
    for _ in range(n - 1):
        line = sys.stdin.readline().split()
        if not line:
            continue  # Handle empty lines if any
        line = list(map(int, line))
        current_j = current_i + 1
        for cost in line:
            A[(current_i, current_j)] = cost
            current_j += 1
        current_i += 1

    min_cost = float('inf')

    # Generate all possible permutations
    for perm in permutations(range(1, n + 1)):
        # Compute the target edge set T
        T = set()
        for (u, v) in g_edges:
            x = perm[u - 1]
            y = perm[v - 1]
            if x < y:
                edge = (x, y)
            else:
                edge = (y, x)
            T.add(edge)
        # Compute symmetric difference and cost
        diff = T.symmetric_difference(h_edges)
        cost = 0
        for edge in diff:
            if edge in A:
                cost += A[edge]
            else:
                # Since edge is in diff, it's either in T or h_edges, but A should have all possible edges
                # However, just in case, but according to problem statement, A contains all i<j
                pass
        if cost < min_cost:
            min_cost = cost

    print(min_cost)

if __name__ == "__main__":
    main()