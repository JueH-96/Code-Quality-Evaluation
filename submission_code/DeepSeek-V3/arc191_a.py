import sys
from math import comb

def solve():
    N, P = map(int, sys.stdin.readline().split())
    max_m = N * (N - 1) // 2
    res = []
    
    # Precompute combinations modulo P up to N choose 2
    # We'll need combinations for various calculations
    
    # The key observation is that the valid graphs must have a BFS tree where the number of vertices at even and odd distances are equal.
    # N is even, so N/2 even and N/2 odd.
    # The BFS tree must be such that the layers alternate between even and odd, starting with 0 (even) for vertex 1.
    
    # The number of such BFS trees is (N/2) * (N/2 - 1)^(N/2 - 1) * (N/2)^(N/2 - 1)
    # But this might not be directly applicable. Instead, we can model the problem as follows:
    # The vertices are split into two sets A and B, each of size N/2. Vertex 1 is in A.
    # The edges in the BFS tree must form a bipartite graph between A and B.
    # The total number of such trees is (N/2)^(N/2 - 1) * (N/2)^(N/2 - 1) * 2? Not sure.
    # According to some references, the number of such trees is (N/2)^{N/2 - 1} * (N/2)^{N/2} * 2^{1 - N/2}? Not sure.
    # Alternatively, for the problem, the number of valid BFS trees is equal to the number of spanning trees of a complete bipartite graph K_{a,b}, where a = b = N/2.
    # The number of spanning trees of K_{a,b} is a^{b-1} * b^{a-1}.
    a = N // 2
    b = N // 2
    num_trees = pow(a, b - 1, P) * pow(b, a - 1, P) % P
    
    # Now, for each M >= N-1, the total number of valid graphs is num_trees multiplied by the number of ways to choose (M - (N-1)) edges from the remaining possible edges, under certain constraints.
    # The remaining edges are those that do not connect two vertices in A or two vertices in B (since that could create a shorter path altering the parity).
    # So the allowed edges are between A and B, but not those already in the spanning tree.
    # The original BFS tree has N-1 edges (all between A and B). The total possible edges between A and B is a*b = (N/2)^2.
    # So, after the tree, there are (a*b - (N-1)) edges left. But the tree uses N-1 edges, all between A and B.
    # So the total possible edges between A and B is a*b. The tree uses N-1 of them. So additional edges must be chosen from the remaining (a*b - (N-1)) edges.
    # But any number of these edges can be added without changing the BFS distances, as long as the graph remains connected (but since the original is a tree, adding any edges won't disconnect it).
    # So for M = N-1 + k, the number of valid graphs is num_trees * C(available, k), where available is a*b - (N-1).
    available = a * b - (N - 1)
    total_edges_ab = a * b
    # So for M = N-1 + k, where 0 <= k <= available, the answer is num_trees * C(available, k)
    # For M > N-1 + available, the answer is 0.
    
    # Also, the total possible edges in the graph is N(N-1)/2. So for M beyond (N-1) + available, the answer is 0.
    # So for each M from N-1 to N(N-1)/2:
    answers = []
    for M in range(N - 1, max_m + 1):
        k = M - (N - 1)
        if k < 0:
            answers.append(0)
            continue
        if k > available:
            answers.append(0)
        else:
            # Compute comb(available, k) mod P
            if available < 0 or k < 0 or k > available:
                c = 0
            else:
                # Precompute factorial, inv factorial up to available mod P
                pass
            # Compute combination on the fly
            c = comb_mod(available, k, P)
            ans = num_trees * c % P
            answers.append(ans)
    
    print(' '.join(map(str, answers)))

def comb_mod(n, k, p):
    if k < 0 or k > n:
        return 0
    numerator = 1
    for i in range(k):
        numerator = numerator * (n - i) % p
    denominator = 1
    for i in range(1, k + 1):
        denominator = denominator * i % p
    return numerator * pow(denominator, p - 2, p) % p

solve()