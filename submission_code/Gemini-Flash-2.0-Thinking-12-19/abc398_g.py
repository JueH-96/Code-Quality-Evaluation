import sys
from collections import deque

# Increase recursion depth for DFS if needed, though BFS is iterative.
# sys.setrecursionlimit(300000)

def n_choose_2_mod_2(x):
    """Calculates (x * (x - 1) // 2) % 2."""
    if x < 2:
        return 0
    # x(x-1)/2 mod 2
    # This is 1 if x % 4 == 2 or x % 4 == 3, and 0 otherwise.
    # x % 4 == 0 => x(x-1)/2 is (4k)(4k-1)/2 = 2k(4k-1) = even
    # x % 4 == 1 => x(x-1)/2 is (4k+1)(4k)/2 = (4k+1)2k = even
    # x % 4 == 2 => x(x-1)/2 is (4k+2)(4k+1)/2 = (2k+1)(4k+1) = 8k^2 + 6k + 1 = odd
    # x % 4 == 3 => x(x-1)/2 is (4k+3)(4k+2)/2 = (4k+3)(2k+1) = 8k^2 + 10k + 3 = odd
    if x % 4 == 2 or x % 4 == 3:
        return 1
    else:
        return 0

def solve():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * (N + 1)
    color = [-1] * (N + 1) # 0 or 1
    components_partition_sizes = [] # List of (size0, size1) tuples for each component

    for i in range(1, N + 1):
        if not visited[i]:
            # Found a new component
            count0 = 0
            count1 = 0
            q = deque([i])
            visited[i] = True
            color[i] = 0
            count0 += 1

            while q:
                u = q.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        color[v] = 1 - color[u]
                        if color[v] == 0:
                            count0 += 1
                        else:
                            count1 += 1
                        q.append(v)
                    # Problem guarantees initial graph is bipartite, no need to check for odd cycles here
                    # else if color[v] == color[u]: graph has odd cycle, problem constraint violated

            components_partition_sizes.append((count0, count1))

    # Calculate the total number of pairs {i, j} with 1 <= i < j <= N
    total_pairs_binom_N_2_mod_2 = n_choose_2_mod_2(N)
    
    # Calculate the total number of "forbidden" pairs {i, j} (i<j)
    # A pair is forbidden if adding an edge creates an odd cycle.
    # This happens iff i and j are in the same component AND in the same partition.
    # Number of such pairs in component k is binom(s_k1, 2) + binom(s_k2, 2).
    # We need the sum modulo 2.
    sum_forbidden_binom_mod_2 = 0
    for s0, s1 in components_partition_sizes:
        sum_forbidden_binom_mod_2 = (sum_forbidden_binom_mod_2 + n_choose_2_mod_2(s0) + n_choose_2_mod_2(s1)) % 2

    # The total number of pairs {i, j} (i<j) where adding an edge is ALLOWED
    # is TotalPairs - ForbiddenPairs.
    # Let this be P_total.
    # P_total = binom(N, 2) - sum_k (binom(s_k1, 2) + binom(s_k2, 2))
    # We need P_total mod 2.
    # (a - b) mod 2 is equivalent to (a + b) mod 2
    P_total_mod_2 = (total_pairs_binom_N_2_mod_2 + sum_forbidden_binom_mod_2) % 2
    
    # The number of available moves is P_total - M (the number of allowed pairs minus existing edges).
    # We need (P_total - M) mod 2.
    # (P_total - M) mod 2 is (P_total mod 2 - M mod 2) mod 2
    # which is (P_total mod 2 + M mod 2) mod 2
    M_mod_2 = M % 2

    total_moves_mod_2 = (P_total_mod_2 + M_mod_2) % 2

    if total_moves_mod_2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

solve()