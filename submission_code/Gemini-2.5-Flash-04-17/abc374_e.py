import sys

def check(C, N, X, processes):
    """
    Checks if a production capacity C is feasible given the budget X.
    """
    if C <= 0:
        return True # Capacity 0 is always feasible with cost 0

    total_cost = 0
    # Use a value guaranteed to be larger than any possible cost for a single process
    # Max cost for one process can be large, use float('inf')
    INF = float('inf')

    for i in range(N):
        A_i, P_i, B_i, Q_i = processes[i]
        
        min_cost_process_i = INF

        # According to a theorem for minimizing ax + by subject to ax + by >= C for integers,
        # an optimal solution exists with x < B or y < A.
        # Here, x is the number of S_i machines (s_i), y is the number of T_i machines (t_i).
        # So we check combinations where s_i in [0, B_i-1] OR t_i in [0, A_i-1].

        # Strategy 1: Iterate s_i from 0 up to B_i - 1
        for s_i in range(B_i): # s_i from 0 to B_i-1
             # Find min t_i >= 0 such that s_i * A_i + t_i * B_i >= C
             # s_i * A_i is the capacity provided by s_i S machines
             capacity_from_s = s_i * A_i
             needed_capacity_from_t = C - capacity_from_s

             t_i = 0
             if needed_capacity_from_t > 0:
                 # Need t_i * B_i >= needed_capacity_from_t
                 # Minimum integer t_i is ceil(needed_capacity_from_t / B_i)
                 # Using integer division formula: (numerator + denominator - 1) // denominator
                 t_i = (needed_capacity_from_t + B_i - 1) // B_i
             # If needed_capacity_from_t <= 0, t_i remains 0. This is correct.

             current_cost = s_i * P_i + t_i * Q_i
             min_cost_process_i = min(min_cost_process_i, current_cost)

        # Strategy 2: Iterate t_i from 0 up to A_i - 1
        for t_i in range(A_i): # t_i from 0 to A_i-1
             # Find min s_i >= 0 such that s_i * A_i + t_i * B_i >= C
             # t_i * B_i is the capacity provided by t_i T machines
             capacity_from_t = t_i * B_i
             needed_capacity_from_s = C - capacity_from_t

             s_i = 0
             if needed_capacity_from_s > 0:
                 # Need s_i * A_i >= needed_capacity_from_s
                 # Minimum integer s_i is ceil(needed_capacity_from_s / A_i)
                 s_i = (needed_capacity_from_s + A_i - 1) // A_i
             # If needed_capacity_from_s <= 0, s_i remains 0. This is correct.

             current_cost = s_i * P_i + t_i * Q_i
             min_cost_process_i = min(min_cost_process_i, current_cost)

        # The minimum cost for process i is the overall minimum found from both strategies
        total_cost += min_cost_process_i

        # Optimization: if total cost already exceeds budget X, we can stop early
        if total_cost > X:
            return False

    # After checking all processes, if total cost is within budget X
    return total_cost <= X


# Read input
N, X = map(int, sys.stdin.readline().split())
processes = []
for _ in range(N):
    A, P, B, Q = map(int, sys.stdin.readline().split())
    processes.append((A, P, B, Q))

# Binary search for the maximum feasible production capacity
# The possible values for capacity C range from 0 upwards.
# A safe upper bound can be estimated. Maximum A_i or B_i is 100.
# If budget X (up to 10^7) is used entirely for one process (e.g., buying machines
# with rate 100 and cost 1), capacity can be around X/1 * 100 = 10^7 * 100 = 10^9.
# Sample 2 confirms 10^9 capacity is possible.
# Set a high enough upper bound, slightly above 10^9.
low = 0
high = 10**9 + 200 # Use 10^9 + a buffer as upper bound
ans = 0 # Stores the maximum feasible capacity found so far

while low <= high:
    mid = (low + high) // 2
    if check(mid, N, X, processes):
        # Capacity `mid` is feasible. It might be the maximum, or a higher capacity might be possible.
        # Store `mid` as a potential answer and try to find a higher one.
        ans = mid
        low = mid + 1
    else:
        # Capacity `mid` is not feasible. The maximum feasible capacity must be lower than `mid`.
        high = mid - 1

# The binary search finds the largest `mid` for which check(mid) returns True.
print(ans)