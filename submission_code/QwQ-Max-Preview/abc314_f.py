MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    parent = list(range(N + 1))
    size = [1] * (N + 1)
    score = [0] * (N + 1)
    inv = [1] * (2 * N + 2)  # Precompute inverses up to 2*N

    # Precompute modular inverses
    for i in range(1, 2 * N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    for _ in range(N - 1):
        p = int(input[idx])
        q = int(input[idx + 1])
        idx += 2
        root_p = find(p)
        root_q = find(q)
        if root_p == root_q:
            continue
        a = size[root_p]
        b = size[root_q]
        # Calculate contributions
        contrib_p = a * inv[a + b] % MOD
        contrib_q = b * inv[a + b] % MOD
        # Add contributions to the roots' scores
        score[root_p] = (score[root_p] + contrib_p) % MOD
        score[root_q] = (score[root_q] + contrib_q) % MOD
        # Merge smaller tree into larger tree
        if size[root_p] < size[root_q]:
            root_p, root_q = root_q, root_p
        parent[root_q] = root_p
        size[root_p] += size[root_q]
        # Propagate the score of root_q to root_p
        score[root_p] = (score[root_p] + score[root_q]) % MOD
        score[root_q] = 0  # Reset the score of the merged team
    
    # Function to propagate scores during find
    def find_and_accumulate(u):
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        # After finding the root, propagate the score to all nodes in the path
        for v in path:
            score[v] = (score[v] + score[u]) % MOD
            parent[v] = u
        return u
    
    # Propagate scores for each node
    for i in range(1, N + 1):
        find_and_accumulate(i)
    
    # Output the results
    print(' '.join(map(str, score[1:N+1])))

if __name__ == '__main__':
    main()