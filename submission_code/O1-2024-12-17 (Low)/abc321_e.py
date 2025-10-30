import sys

def main():
    input_data = sys.stdin.read().strip().split()
    T = int(input_data[0])
    idx = 1
    
    out = []
    
    # A small helper: counts how many descendants are exactly distance d down from node A,
    # in the infinite full binary tree, but truncated at node index N.
    def subtree_count(A, d, N):
        # If we're asked for negative distance or invalid distance, no nodes.
        if d < 0:
            return 0
        
        # Leftmost node at distance d is A << d
        L = A << d
        if L > N:
            return 0
        
        # Rightmost node at distance d (if unbounded) is ((A+1) << d) - 1
        R = ((A + 1) << d) - 1
        if R < L:
            return 0
        
        # Truncate R by N
        R = min(R, N)
        return max(0, R - L + 1)
    
    for _ in range(T):
        N = int(input_data[idx]); idx+=1
        X = int(input_data[idx]); idx+=1
        K = int(input_data[idx]); idx+=1
        
        # If K == 0, answer is simply 1 (the node X itself), provided X<=N (by constraints it is).
        # But let's handle in the general logic below anyway.
        
        # The maximum possible distance in this "binary-tree" shape is about 2 * (floor(log2(N))).
        # If K is greater than that, the answer is 0 immediately.
        dN = N.bit_length() - 1  # roughly the depth of the largest node
        if K > 2*dN:
            out.append('0')
            continue
        
        # We'll sum over how many ways we can go "u" steps up from X and then (K-u) steps down
        # in the subtree of that ancestor (excluding the child-subtree that leads back to X).
        dX = X.bit_length() - 1  # depth of X in this tree
        # We can't go up more than dX steps (that would reach the root).
        max_up = min(K, dX)
        
        ans = 0
        for u in range(max_up + 1):
            # A = ancestor of X by u steps
            # in this tree, going up u steps = floor(X / 2^u) = X >> u
            A = X >> u
            d_down = K - u
            
            # Count all nodes that are distance d_down below A (in the subtree), truncated by N.
            cntA = subtree_count(A, d_down, N)
            
            # Exclude the subtree going back to X if we actually went up (u>0) and we still have
            # to go some steps down (d_down>0).  Because going back down that same edge
            # would not form a simple path of distance K from X -- that would "reverse" the edge.
            if u > 0 and d_down > 0:
                # The child c of A that lies on the path back to X is the ancestor of X that is (u-1) steps up
                c = X >> (u - 1)
                cntC = subtree_count(c, d_down - 1, N)
                cntA -= cntC
            
            ans += cntA
        
        out.append(str(ans))
    
    print('
'.join(out))

# Don't forget to call main().
if __name__ == "__main__":
    main()