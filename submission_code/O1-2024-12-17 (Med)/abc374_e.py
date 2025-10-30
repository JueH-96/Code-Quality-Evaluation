def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = int(input_data[1])
    
    # Machines data: A_i, P_i, B_i, Q_i
    machines = []
    idx = 2
    for _ in range(N):
        A = int(input_data[idx]);   idx += 1
        P = int(input_data[idx]);   idx += 1
        B = int(input_data[idx]);   idx += 1
        Q = int(input_data[idx]);   idx += 1
        machines.append((A, P, B, Q))
        
    # Function to compute the minimum cost for process (A, P, B, Q) to reach capacity >= M
    def cost_to_achieve(M, A, P, B, Q):
        # If we want a capacity of 0, no machines are needed -> cost 0
        if M <= 0:
            return 0
        
        # We'll try x in [0..B-1] (because B <= 100) and compute
        # how many of the B-machines are needed to reach capacity >= M.
        min_cost = float('inf')
        for x in range(B):
            # capacity from x units of S-machine
            cap = x * A
            if cap >= M:
                # no need for T-machine
                cost = x * P
            else:
                # how many B-machines needed
                needed = M - cap
                y = (needed + B - 1) // B  # ceil division
                cost = x * P + y * Q
            if cost < min_cost:
                min_cost = cost
        return min_cost
    
    # Check if we can achieve production capacity >= mid within budget X
    def can_achieve(mid):
        total_cost = 0
        for (A, P, B, Q) in machines:
            c = cost_to_achieve(mid, A, P, B, Q)
            total_cost += c
            # Early stop if we exceed budget
            if total_cost > X:
                return False
        return total_cost <= X
    
    # Binary search for the maximum M (production capacity) from 0..1e9
    low, high = 0, 10**9 + 1
    while low + 1 < high:
        mid = (low + high) // 2
        if can_achieve(mid):
            low = mid
        else:
            high = mid
    
    print(low)

# Do not forget to call main()
main()