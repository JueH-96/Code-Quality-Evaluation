def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    MOD = 998244353
    
    # Helper function to compute modular inverse using Fermat's little theorem
    # Since MOD is prime, inv(x) = x^(MOD-2) mod MOD
    def inv(x):
        return pow(x, MOD - 2, MOD)
    
    # Read inputs
    N = int(input_data[0])
    p = [0]*(N-1)
    q = [0]*(N-1)
    idx = 1
    for i in range(N-1):
        p[i] = int(input_data[idx]); q[i] = int(input_data[idx+1])
        idx += 2
    
    # Disjoint Set Union (Union-Find) with
    # "setVal[root]" = common offset for entire set (root)
    # "offset[x]" = offset from x to its root
    parent = list(range(N+1))
    size = [1]*(N+1)
    setVal = [0]*(N+1)
    offset = [0]*(N+1)  # offset[x]: partial offset from x upwards toward the root
    
    def find(u):
        """ Find the root of u with path compression, 
            accumulating offset from u to the (new) root. """
        if parent[u] != u:
            pu = parent[u]
            parent[u] = find(pu)       # recursively find the ultimate root
            offset[u] = (offset[u] + offset[pu]) % MOD
        return parent[u]
    
    def union(u, v):
        """ Merge the sets containing u and v, given the specific
            probabilities that the set of u or the set of v wins. """
        ru = find(u)
        rv = find(v)
        if ru == rv:
            return
        # Probability that ru's team wins: size[ru] / (size[ru] + size[rv])
        # Probability that rv's team wins: size[rv] / (size[ru] + size[rv])
        su = size[ru]
        sv = size[rv]
        denom_inv = inv(su + sv)  # inverse of (su + sv)
        p_u = (su * denom_inv) % MOD
        p_v = (sv * denom_inv) % MOD
        
        # Add p_u to the entire set ru, and p_v to the entire set rv
        setVal[ru] = (setVal[ru] + p_u) % MOD
        setVal[rv] = (setVal[rv] + p_v) % MOD
        
        # Union by size (attach the smaller root to the bigger root)
        if size[ru] < size[rv]:
            ru, rv = rv, ru  # swap, so ru is the bigger set
        
        # Now attach rv -> ru
        parent[rv] = ru
        # offset[rv] becomes setVal[rv] - setVal[ru], so that for any node x in rv's subtree:
        #    total offset = offset[x] + offset[rv] + setVal[ru]
        # but setVal[rv] + offset[x] was that subtree's old total => to keep continuity,
        # offset[rv] = setVal[rv] - setVal[ru].
        offset[rv] = (setVal[rv] - setVal[ru]) % MOD
        
        # The root rv is no longer a root, so we can setVal[rv] = 0, not strictly necessary
        setVal[rv] = 0
        
        size[ru] += size[rv]
    
    # Process matches in the given order
    for i in range(N-1):
        union(p[i], q[i])
    
    # Final expected values: for each i, E[i] = setVal[ root(i) ] + offset[i]
    # where root(i) = find(i).
    # We compute them and output.
    result = []
    for i in range(1, N+1):
        r = find(i)
        Ei = setVal[r] + offset[i]
        Ei %= MOD
        result.append(str(Ei))
    
    print(" ".join(result))