def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    T = int(input_data[0])
    
    # Precompute a "large" value to use when 2^h would exceed 10^18
    # (1<<60) is 1152921504606846976, which is > 10^18.
    # We'll use this as a stand-in for "an effectively infinite power of 2".
    INF_P2 = 1 << 60
    
    idx = 1
    
    # A small function to count the number of nodes exactly dist h down
    # from a node Y (in the "infinite" binary tree) subject to node <= N.
    # The set of nodes at distance h from Y is all integers from (Y << h)
    # to (Y << h) + 2^h - 1, intersected with [1..N].
    # So the count is min(2^h, N - (Y << h) + 1) if (Y << h) <= N, else 0.
    #
    # We'll implement it inline in the loop to avoid function-call overhead.
    
    out = []
    for _ in range(T):
        N = int(input_data[idx]); idx+=1
        X = int(input_data[idx]); idx+=1
        K = int(input_data[idx]); idx+=1
        
        # Depth of X in this tree = number of times we can do X//=2 till X=1
        # A faster way in Python is bit_length - 1.
        dx = X.bit_length() - 1
        
        # We only need to consider d up to min(dx, K),
        # because going "up" more steps than dx is impossible (above the root),
        # and obviously we won't go beyond K either.
        max_d = min(dx, K)
        
        ans = 0
        
        # Loop over how many steps we go "up" from X
        # Then we go "down" (K-d) steps in the subtree of that ancestor.
        for d in range(max_d + 1):
            # The ancestor after going d steps up is simply X >> d
            Y = X >> d
            h = K - d  # how many steps (edges) to go down from Y
            
            # First check if it's even possible that Y<<h <= N
            # If h >= 60, then 2^h >= 2^60 ~ 1.15e18 > 1e18 (which is >= N).
            # We'll do a direct check using Python's arbitrary precision,
            # but skip computing 2^h directly if h >= 60.
            #
            # Condition for having any nodes in that down-subtree:
            #   (Y << h) <= N
            # If that holds, the count is min(2^h, N - (Y<<h) + 1).
            
            # Quick check to avoid big shifts if Y == 0, etc.
            if Y == 0:
                # Means X < 2^d, so we can't actually go up d steps
                # (in practice X>>d would become 0). No further d will work, so break.
                break
            
            # Check if (Y << h) <= N
            # We'll do Y <= (N >> h) as a safer comparison when h < 60.
            if h < 60:
                if Y <= (N >> h):
                    # Then (Y << h) <= N
                    val = N - (Y << h) + 1
                    p2h = (1 << h)  # 2^h
                    ans += p2h if p2h < val else val
            else:
                # h >= 60 => 2^h definitely >= 2^60 > 1e18 >= N
                # So 2^h >= N+something, the min(2^h, X) part is effectively
                # controlled by (N - (Y<<h) + 1) if (Y<<h) <= N.
                # We check (Y<<h) <= N by direct shift:
                # (Python handles big integers safely.)
                if (Y << h) <= N:
                    val = N - (Y << h) + 1
                    ans += val
        
        out.append(str(ans))
    
    print("
".join(out))