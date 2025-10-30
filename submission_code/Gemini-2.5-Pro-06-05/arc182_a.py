import sys

# It's recommended to increase recursion limit for deep DSU structures.
sys.setrecursionlimit(2 * 10**5)

def solve():
    """
    Solves the problem by modeling it as a system of pairwise constraints
    and using a Disjoint Set Union (DSU) data structure to count solutions.
    """
    try:
        input = sys.stdin.readline
        N, Q = map(int, input().split())
        queries = [tuple(map(int, input().split())) for _ in range(Q)]
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    MOD = 998244353

    # DSU for Q operations + 1 dummy node for TRUE (representing the 'prefix' choice).
    # The dummy node is at index Q.
    parent = list(range(Q + 1))
    # diff[i] = 0 means choice[i] and choice[parent[i]] are the same (both prefix or both suffix).
    # diff[i] = 1 means they are different.
    # We define choice=0 for prefix, choice=1 for suffix.
    diff = [0] * (Q + 1)

    def find(i):
        if parent[i] == i:
            return i, 0
        root, parity = find(parent[i])
        parent[i] = root
        diff[i] ^= parity
        return root, diff[i]

    def unite(i, j, d):
        # Enforce choice[i] ^ choice[j] == d
        root_i, parity_i = find(i)
        root_j, parity_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            diff[root_j] = parity_i ^ parity_j ^ d
            return True
        else:
            # If they are already in the same set, check for contradiction.
            if (parity_i ^ parity_j) != d:
                return False
            return True

    TRUE_NODE = Q

    for i in range(Q):
        for j in range(i + 1, Q):
            Pi, Vi = queries[i]
            Pj, Vj = queries[j]

            # The problem defines constraints for k < j with V_k > V_j.
            # Here, i is always less than j.
            if Vi > Vj:
                if Pi == Pj:
                    print(0)
                    return
                if Pi < Pj:  # Must be (C_i, C_j) = (prefix, suffix)
                    # choice[i]=0, choice[j]=1
                    # choice[i] ^ choice[TRUE_NODE] = 0
                    # choice[j] ^ choice[TRUE_NODE] = 1
                    if not unite(i, TRUE_NODE, 0): print(0); return
                    if not unite(j, TRUE_NODE, 1): print(0); return
                else:  # Pi > Pj, must be (C_i, C_j) = (suffix, prefix)
                    # choice[i]=1, choice[j]=0
                    if not unite(i, TRUE_NODE, 1): print(0); return
                    if not unite(j, TRUE_NODE, 0): print(0); return
            elif Vj > Vi:
                # Symmetric case: here op j has larger V, so it's the 'k' in the rule.
                if Pj == Pi:
                    print(0)
                    return
                if Pj < Pi:  # Must be (C_j, C_i) = (prefix, suffix)
                    # choice[j]=0, choice[i]=1
                    if not unite(j, TRUE_NODE, 0): print(0); return
                    if not unite(i, TRUE_NODE, 1): print(0); return
                else:  # Pj > Pi, must be (C_j, C_i) = (suffix, prefix)
                    # choice[j]=1, choice[i]=0
                    if not unite(j, TRUE_NODE, 1): print(0); return
                    if not unite(i, TRUE_NODE, 0): print(0); return

    # Count components not connected to TRUE_NODE.
    # Each such component has 2 independent choices.
    num_free_components = 0
    root_true, _ = find(TRUE_NODE)
    for i in range(Q):
        if parent[i] == i:
            if i != root_true:
                num_free_components += 1
    
    ans = pow(2, num_free_components, MOD)
    print(ans)

solve()