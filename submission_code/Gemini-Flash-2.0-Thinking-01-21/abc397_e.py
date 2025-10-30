import sys
sys.setrecursionlimit(300000)

def solve():
    N, K = map(int, sys.stdin.readline().split())
    NK = N * K

    if K == 1:
        # Any tree with N vertices can be decomposed into N paths of length 1 (single vertices).
        # The input guarantees a tree with NK = N vertices.
        # Condition 1: P is permutation of 1..N. Trivial, P_i,1 = i.
        # Condition 2: Edges between P_i,j and P_i,j+1 for j=1..K-1. K-1=0, vacuously true.
        # So for K=1, the answer is always Yes.
        # Read edges to consume input
        for _ in range(NK - 1):
            sys.stdin.readline()
        print("Yes")
        return

    adj = [[] for _ in range(NK + 1)]
    for _ in range(NK - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # dp[u]: size of path fragment ending at u, 0 <= dp[u] < K.
    # dp[u] = 0 means subtree at u is perfectly decomposed into paths of size K.
    # dp[u] = -1 means impossible.

    def dfs(u, p):
        counts = [0] * K # counts[s] = number of children v with dp[v] = s
        
        is_leaf = True
        for v in adj[u]:
            if v != p:
                is_leaf = False
                res = dfs(v, u)
                if res == -1:
                    return -1 # Impossible in child subtree
                # res is in {0, 1, ..., K-1}
                counts[res] += 1

        if is_leaf:
             return 1 # Leaf forms a fragment of size 1

        # Now combine fragments from children. Fragments with size s > 0 are active.
        # We can pair s1 and s2 if s1 + s2 + 1 = K. s1 + s2 = K - 1.

        # Use a list to store remaining fragment sizes
        remaining_fragments_list = []

        # Iterate through possible fragment sizes s from 1 up to (K-1)/2
        for s in range(1, (K + 1) // 2):
            p = min(counts[s], counts[K - 1 - s])
            remaining_fragments_list.extend([s] * (counts[s] - p))
            remaining_fragments_list.extend([K - 1 - s] * (counts[K - 1 - s] - p))

        # If K-1 is even, handle the middle element s = (K-1)/2
        if (K - 1) % 2 == 0:
             s0 = (K - 1) // 2
             remaining_fragments_list.extend([s0] * (counts[s0] % 2))
        
        remaining_count = len(remaining_fragments_list)

        if remaining_count > 1:
            return -1 # Cannot form a single upward-going fragment

        if remaining_count == 1:
            # One fragment of size s* remains. u joins it.
            s_star = remaining_fragments_list[0]
            result_size = s_star + 1
            if result_size == K:
                return 0 # Forms a complete path of size K
            else:
                return result_size # Fragment of size < K connects upwards

        # remaining_count == 0: All positive fragments are paired up.
        # u must form a fragment of size 1 extending upwards.
        return 1 # Fragment size 1 connects upwards

    # Start DFS from vertex 1, parent 0.
    # The root cannot connect upwards. The fragment size at the root must be 0.
    # The logic inside DFS assumes u might need to connect upwards (result 1).
    # If the DFS on the root returns > 0, it means a fragment of that size remains at the root, which is impossible.
    # So only result 0 from dfs(1,0) is Yes.

    root_result = dfs(1, 0)

    if root_result == 0:
        print("Yes")
    else:
        print("No")

solve()