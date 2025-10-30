def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    X = int(data[1])
    
    A = [0]*N
    P = [0]*N
    B = [0]*N
    Q = [0]*N
    
    idx = 2
    for i in range(N):
        A[i] = int(data[idx]); idx += 1
        P[i] = int(data[idx]); idx += 1
        B[i] = int(data[idx]); idx += 1
        Q[i] = int(data[idx]); idx += 1
    
    # Precompute a function to get minimal cost for process i to reach capacity >= K
    # We use an O(A_i + B_i) "around the best guess" approach to handle integer mixes
    def cost_for_process(K, Ai, Pi, Bi, Qi):
        # If K == 0, no machines needed -> cost 0
        if K == 0:
            return 0
        
        best_cost = float('inf')
        
        # Compare Pi/Bi vs Qi/Ai by cross multiplication to avoid float
        # d < 0 => S is "cheaper" in continuous sense
        # d > 0 => T is "cheaper"
        # d = 0 => same ratio
        d = Pi * Bi - Qi * Ai
        
        def check_s_range(s_start, s_end):
            nonlocal best_cost
            for s in range(max(0, s_start), max(0, s_end)+1):
                leftover = K - Ai*s
                if leftover > 0:
                    t = (leftover + Bi - 1) // Bi
                    cost = Pi*s + Qi*t
                else:
                    cost = Pi*s
                if cost < best_cost:
                    best_cost = cost
        
        def check_t_range(t_start, t_end):
            nonlocal best_cost
            for t in range(max(0, t_start), max(0, t_end)+1):
                leftover = K - Bi*t
                if leftover > 0:
                    s = (leftover + Ai - 1) // Ai
                    cost = Pi*s + Qi*t
                else:
                    cost = Qi*t
                if cost < best_cost:
                    best_cost = cost
        
        if d < 0:
            # S is better ratio
            s_best = (K + Ai - 1) // Ai
            check_s_range(s_best - Bi, s_best + Bi)
        
        elif d > 0:
            # T is better ratio
            t_best = (K + Bi - 1) // Bi
            check_t_range(t_best - Ai, t_best + Ai)
        
        else:
            # Same ratio; try both ways
            s_best = (K + Ai - 1) // Ai
            check_s_range(s_best - Bi, s_best + Bi)
            t_best = (K + Bi - 1) // Bi
            check_t_range(t_best - Ai, t_best + Ai)
        
        return best_cost
    
    # Check feasibility of capacity = cap
    def can(cap):
        total_cost = 0
        for i in range(N):
            c = cost_for_process(cap, A[i], P[i], B[i], Q[i])
            total_cost += c
            if total_cost > X:
                return False
        return total_cost <= X
    
    # Binary search for maximum feasible capacity
    lo, hi = 0, 10**9 + 1  # hi is set just above the largest plausible capacity
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if can(mid):
            lo = mid
        else:
            hi = mid
    
    print(lo)

# Don't forget to call main()
if __name__ == "__main__":
    main()