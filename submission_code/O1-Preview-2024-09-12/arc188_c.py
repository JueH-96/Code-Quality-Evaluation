# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())

    # Union-Find with Parity

    parent = [i for i in range(N+1)]
    rank = [0] * (N+1)
    parity = [0] * (N+1)  # parity[u]: parity from u to parent[u]

    def find(u):
        if parent[u] != u:
            orig_parent = parent[u]
            parent[u] = find(parent[u])
            parity[u] ^= parity[orig_parent]
        return parent[u]

    def union(u, v, w):
        pu = find(u)
        pv = find(v)
        total_parity = parity[u] ^ parity[v] ^ w
        if pu != pv:
            # Union by rank can be added if needed
            parent[pu] = pv
            parity[pu] = total_parity
            return True
        else:
            if total_parity != 0:
                # Conflict detected
                return False
            else:
                return True

    # Build equations
    from collections import defaultdict
    testimonies_by_villager = defaultdict(list)  # villager u: list of (v, C_i)
    all_villagers_who_testified = set()

    for _ in range(M):
        A_i, B_i, C_i = map(int, sys.stdin.readline().split())
        all_villagers_who_testified.add(A_i)
        testimonies_by_villager[A_i].append((B_i, C_i))

    equations = []
    # For each villager who gives testimonies, generate equations between h_v
    for u in testimonies_by_villager:
        lst = testimonies_by_villager[u]
        n = len(lst)
        # For each pair of testimonies from u, generate an equation
        for i in range(n):
            v1, C1 = lst[i]
            for j in range(i+1, n):
                v2, C2 = lst[j]
                w = C1 ^ C2  # w = C1 XOR C2
                # Equation: h_v1 XOR h_v2 == w
                res = union(v1, v2, w)
                if not res:
                    print(-1)
                    return

    # Assign h_u arbitrarily for one node per connected component
    NODES = N
    h = [0] * (N+1)  # h[1..N], h[0] unused
    assigned = [False] * (N+1)
    for u in range(1, N+1):
        find(u)  # Ensure path compression
    for u in range(1, N+1):
        pu = parent[u]
        if not assigned[pu]:
            assigned[pu] = True
            h[pu] = 0  # Assign h[pu] = 0 arbitrarily

    # Now, compute h[u]
    for u in range(1, N+1):
        pu = parent[u]
        h[u] = h[pu] ^ parity[u]

    # Now compute c_u for all villagers who testified
    c = [0] * (N+1)
    for u in testimonies_by_villager:
        # Use one of the testimonies to compute c_u
        v, C_i = testimonies_by_villager[u][0]
        c_u = h[u] ^ h[v] ^ C_i
        c[u] = c_u

    # For villagers who didn't testify, assign c_u arbitrarily (e.g., 0)
    for u in range(1, N+1):
        if u not in all_villagers_who_testified:
            c[u] = 0  # Assign arbitrarily

    # Output
    result = ''.join(map(str, c[1:]))
    print(result)

threading.Thread(target=main).start()