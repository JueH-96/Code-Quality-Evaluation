import math

def solve():
    N, X = map(int, input().split())
    processes_params = []
    for _ in range(N):
        processes_params.append(list(map(int, input().split())))

    # Memoization dictionary for (A, P, B, Q, capacity_C) -> min_cost
    # This helps if multiple processes have identical parameters or if check(C) is somehow called with the same C multiple times
    # for the same process configuration (though binary search structure makes the latter unlikely for C).
    # Primarily useful if some (A,P,B,Q) tuples are repeated among processes.
    memo_min_cost_process_config = {}

    def get_min_cost_for_process_config(A, P, B, Q, capacity_C):
        state = (A, P, B, Q, capacity_C)
        if state in memo_min_cost_process_config:
            return memo_min_cost_process_config[state]

        if capacity_C == 0: # 0 capacity requires 0 cost
            return 0
        
        min_total_cost_for_this_process = float('inf')

        # Option 1: Iterate number of S machines (s_count), calculate required T machines.
        # Iterate s_count from 0 up to B (inclusive). Max B is 100.
        # The loop runs B+1 times.
        limit_s_iterations = B 
        for s_count in range(limit_s_iterations + 1): 
            cost_s_machines = s_count * P
            products_from_s = s_count * A
            
            remaining_products_needed = capacity_C - products_from_s
            
            cost_t_machines = 0
            if remaining_products_needed > 0:
                # t_count = math.ceil(remaining_products_needed / B)
                # Integer arithmetic for ceil: (numerator + denominator - 1) // denominator
                t_count = (remaining_products_needed + B - 1) // B
                cost_t_machines = t_count * Q
            
            current_combination_cost = cost_s_machines + cost_t_machines
            min_total_cost_for_this_process = min(min_total_cost_for_this_process, current_combination_cost)

        # Option 2: Iterate number of T machines (t_count), calculate required S machines.
        # Iterate t_count from 0 up to A (inclusive). Max A is 100.
        # The loop runs A+1 times.
        limit_t_iterations = A
        for t_count in range(limit_t_iterations + 1):
            cost_t_machines = t_count * Q
            products_from_t = t_count * B

            remaining_products_needed = capacity_C - products_from_t

            cost_s_machines = 0
            if remaining_products_needed > 0:
                # s_count = math.ceil(remaining_products_needed / A)
                s_count = (remaining_products_needed + A - 1) // A
                cost_s_machines = s_count * P
            
            current_combination_cost = cost_s_machines + cost_t_machines
            min_total_cost_for_this_process = min(min_total_cost_for_this_process, current_combination_cost)
        
        memo_min_cost_process_config[state] = min_total_cost_for_this_process
        return min_total_cost_for_this_process

    def check(target_capacity):
        if target_capacity == 0:
            return True 
            
        current_total_budget_spent = 0
        for i in range(N):
            A_i, P_i, B_i, Q_i = processes_params[i]
            cost_for_this_process = get_min_cost_for_process_config(A_i, P_i, B_i, Q_i, target_capacity)
            current_total_budget_spent += cost_for_this_process
            
            if current_total_budget_spent > X: # Early exit if budget exceeded
                return False
        
        return current_total_budget_spent <= X

    low = 0
    high = 2 * (10**9)  # A safe upper bound for capacity C
    ans = 0

    # Binary search for maximum C
    # log2(2*10^9) is about 31. 100 iterations are very safe.
    for _ in range(100): 
        if low > high:
            break
        mid = low + (high - low) // 2
        
        if check(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
            
    print(ans)

solve()