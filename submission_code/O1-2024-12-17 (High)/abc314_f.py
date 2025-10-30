def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    
    # Modulus
    MOD = 998244353
    
    # Read N
    N = int(input_data[0])
    
    # p_i, q_i pairs (convert to 0-based)
    p = [int(input_data[2*i+1]) - 1 for i in range(N-1)]
    q = [int(input_data[2*i+2]) - 1 for i in range(N-1)]
    
    # Build inverses up to N (for 1..N), so we can do "x * inv[y]" mod
    inv = [0] * (N+1)
    inv[1] = 1
    for i in range(2, N+1):
        inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD
    
    # DSU (Disjoint Set Union) with "small-to-large" merging
    parent = list(range(N))
    size = [1]*N
    
    # For storing each player's expected number of wins
    eVal = [0]*N
    
    # "members[r]" will hold the list of actual nodes in the DSU component whose leader is r
    # Initially each node is alone in its own component
    members = [[i] for i in range(N)]
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    # Process matches in the given order
    # The side containing p[i] is "first" and gets + size(p[i])/(size(p[i]) + size(q[i]))
    # The side containing q[i]) is "second" and gets + size(q[i])/(size(p[i]) + size(q[i]))
    for i in range(N-1):
        px = find(p[i])
        qx = find(q[i])
        if px == qx:
            # Should never happen by problem statement, but just in case
            continue
        
        # Sizes of these two components
        a = size[px]
        b = size[qx]
        total = a + b
        
        # Probabilities (in modular form)
        # first side (px) wins with probability a/(a+b); second side (qx) with probability b/(a+b)
        # Each member of px gets + a/(a+b), each member of qx gets + b/(a+b)
        prob_px = (a * inv[total]) % MOD
        prob_qx = (b * inv[total]) % MOD
        
        # Add increments
        for node in members[px]:
            eVal[node] = (eVal[node] + prob_px) % MOD
        for node in members[qx]:
            eVal[node] = (eVal[node] + prob_qx) % MOD
        
        # Union by size (attach smaller to bigger)
        if size[px] < size[qx]:
            px, qx = qx, px  # swap so px is always the bigger
        # Now px is the leader of the merged component
        parent[qx] = px
        size[px] += size[qx]
        
        # Merge the node-lists
        members[px].extend(members[qx])
        members[qx].clear()
    
    # Print results for players 1..N (stored as 0..N-1)
    # They are already in the required modular form
    print(" ".join(map(str, eVal)))
    
# Do not forget to call main()!
main()