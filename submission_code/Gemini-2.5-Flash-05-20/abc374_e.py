import sys

def solve():
    N, X = map(int, sys.stdin.readline().split())
    
    processes = []
    for _ in range(N):
        A, P, B, Q = map(int, sys.stdin.readline().split())
        processes.append((A, P, B, Q))

    # Function to check if a production capacity K is achievable within budget X
    def check(K):
        if K == 0:
            return True # 0 capacity costs 0, always possible
        
        total_cost = 0
        for i in range(N):
            A_i, P_i, B_i, Q_i = processes[i]
            
            min_cost_for_i = float('inf') # Initialize with a very large value

            # The key optimization for finding min_cost_for_i:
            # An optimal solution (s_count, t_count) for s_count * A_i + t_count * B_i >= K
            # will have either s_count < B_i/gcd(A_i, B_i) or t_count < A_i/gcd(A_i, B_i).
            # Since A_i, B_i are at most 100, the maximum value for B_i/gcd or A_i/gcd is 100.
            # So, we iterate s_count from 0 to 100 and find required t_count, and vice versa.

            # Strategy 1: Iterate s_count (units of S_i), cover remaining capacity with T_i
            # Loop s_count from 0 up to 100 (inclusive)
            for s_count in range(101): 
                current_cap_s = s_count * A_i
                current_cost_s = s_count * P_i
                
                remaining_cap_needed = K - current_cap_s
                
                if remaining_cap_needed <= 0:
                    # Capacity K is already met or exceeded by S_i machines alone
                    min_cost_for_i = min(min_cost_for_i, current_cost_s)
                else:
                    # Need to cover remaining_cap_needed with T_i machines
                    # Calculate required t_count using ceiling division
                    # (x + y - 1) // y gives ceil(x/y) for positive x, y
                    t_count = (remaining_cap_needed + B_i - 1) // B_i 
                    total_current_cost = current_cost_s + t_count * Q_i
                    min_cost_for_i = min(min_cost_for_i, total_current_cost)

            # Strategy 2: Iterate t_count (units of T_i), cover remaining capacity with S_i
            # Loop t_count from 0 up to 100 (inclusive)
            for t_count in range(101):
                current_cap_t = t_count * B_i
                current_cost_t = t_count * Q_i
                
                remaining_cap_needed = K - current_cap_t
                
                if remaining_cap_needed <= 0:
                    # Capacity K is already met or exceeded by T_i machines alone
                    min_cost_for_i = min(min_cost_for_i, current_cost_t)
                else:
                    # Need to cover remaining_cap_needed with S_i machines
                    s_count = (remaining_cap_needed + A_i - 1) // A_i
                    total_current_cost = current_cost_t + s_count * P_i
                    min_cost_for_i = min(min_cost_for_i, total_current_cost)
            
            # Add the minimum cost for current process to total cost
            total_cost += min_cost_for_i
            
            # Optimization: If total_cost already exceeds budget X, no need to continue
            if total_cost > X:
                return False
                
        # If all processes can achieve K capacity within budget
        return total_cost <= X

    # Binary search for the maximum K
    low = 0
    # A safe upper bound for K: max X (10^7) / min cost (1) * max capacity (100) = 10^9
    # Add a small buffer just to be super safe.
    high = 2 * 10**9 
    ans = 0 # Stores the maximum achievable K

    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            ans = mid         # mid is achievable, try for a higher K
            low = mid + 1
        else:
            high = mid - 1    # mid is too high, try for a lower K
            
    sys.stdout.write(str(ans) + '
')

solve()