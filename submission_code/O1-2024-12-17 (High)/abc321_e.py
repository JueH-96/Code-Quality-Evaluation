def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    T = int(input_data[0])
    
    # We'll process the input in chunks of three after the first line
    idx_out = []
    pos = 1
    
    # A small helper (inlined) to count how many nodes lie at distance d
    # from node u in the subtree, given the maximum label N.
    # The set of nodes at distance d from u in an infinite complete binary
    # tree would be [u << d, (u << d) + 2^d - 1], clamped by N.
    # We also special-case large d to avoid huge shifts.
    def count_down(u, d, N):
        # distance can't be negative
        if d < 0:
            return 0
        # if d is too large, 2^d surely exceeds N (since N <= 10^18 < 2^60)
        if d > 59:
            return 0
        L = u << d
        if L > N:
            return 0
        R = L + (1 << d) - 1
        if R > N:
            return max(0, N - L + 1)
        return (1 << d)
    
    # Solve each test case
    out = []
    for _ in range(T):
        N = int(input_data[pos]); X = int(input_data[pos+1]); K = int(input_data[pos+2])
        pos += 3
        
        # Build the ancestor chain from X up to 1:
        anc = []
        v = X
        while True:
            anc.append(v)
            if v == 1:
                break
            v //= 2
        # D is how many steps from X up to the root
        D = len(anc) - 1
        
        total = 0
        
        # We only need to go up min(K, D) steps
        # a = number of steps going up from X to some ancestor A
        # b = K - a = then the distance we must go down from A
        # if b==0, that means the node A itself is distance K
        # if b>0, we go into children of A not leading back to X.
        limit = min(K, D)
        for a in range(limit+1):
            A = anc[a]
            b = K - a
            
            if b == 0:
                # purely "up" path
                total += 1
            else:
                # consider children of A that are not the one on the path back to X
                # A might have up to two children: 2*A and 2*A+1 (if <= N)
                # if a>0, then anc[a-1] is the child pointing back to X
                blocked_child = anc[a-1] if a > 0 else -1
                
                c1 = 2 * A
                if c1 <= N and c1 != blocked_child:
                    # We want distance (b-1) in that subtree
                    total += count_down(c1, b-1, N)
                
                c2 = c1 + 1
                if c2 <= N and c2 != blocked_child:
                    total += count_down(c2, b-1, N)
        
        out.append(str(total))
    
    # Print all answers
    print("
".join(out))

# Do not forget to call main()
main()