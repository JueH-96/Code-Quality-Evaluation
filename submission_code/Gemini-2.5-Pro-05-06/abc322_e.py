import math
import sys

def main():
    N, K, P = map(int, sys.stdin.readline().split())
    
    plans_data = []
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        cost = line[0]
        params_increase = tuple(line[1:])
        plans_data.append({'cost': cost, 'params_increase': params_increase})

    P_plus_1 = P + 1 # Value P+1 used for base in multi-dimensional index mapping

    # dp[idx] stores the minimum cost to reach the state represented by idx.
    # A state is a K-tuple (p1, p2, ..., pK) where 0 <= pj <= P.
    num_states = P_plus_1**K
    dp = [math.inf] * num_states

    # Helper function to convert a K-tuple of parameter values to a 1D index.
    def get_idx_from_params(param_values_tuple):
        idx = 0
        for val in param_values_tuple:
            idx = idx * P_plus_1 + val
        return idx

    # Helper function to convert a 1D index back to a K-tuple of parameter values.
    # This is called for each state in each plan iteration.
    # Can be memoized or simply recomputed; K is small.
    # For this version, it's recomputed.
    params_cache = {} # Cache for get_params_from_idx
    def get_params_from_idx(idx_val):
        if idx_val in params_cache:
            return params_cache[idx_val]

        params_list = [0] * K
        temp_idx_val = idx_val 
        for j in range(K - 1, -1, -1): # Fill from right to left
            params_list[j] = temp_idx_val % P_plus_1
            temp_idx_val //= P_plus_1
        
        result_tuple = tuple(params_list)
        params_cache[idx_val] = result_tuple
        return result_tuple

    # Initial state: all parameters are 0. Cost is 0.
    initial_params_tuple = tuple([0] * K)
    dp[get_idx_from_params(initial_params_tuple)] = 0
    
    # Populate cache for (0,0,...,0) as it might be idx 0.
    if K > 0 : # only if K > 0, otherwise initial_params_tuple is empty. It won't be empty by constraints.
         params_cache[get_idx_from_params(initial_params_tuple)] = initial_params_tuple


    # Iterate through each development plan
    for i in range(N):
        current_plan_cost = plans_data[i]['cost']
        current_plan_A = plans_data[i]['params_increase']

        # Iterate through all possible source states S = (s1, ..., sK)
        # Iterate current_idx from num_states-1 down to 0.
        for current_idx in range(num_states - 1, -1, -1):
            if dp[current_idx] == math.inf:
                # This state S is not reachable with plans processed so far.
                continue

            # current_params_s_tuple is state S, reached with cost dp[current_idx]
            current_params_s_tuple = get_params_from_idx(current_idx)
            
            # Calculate the new state S_new if we apply the current plan i
            # new_sj = min(P, sj + A_ij)
            new_params_s_prime_list = [0] * K
            for j in range(K):
                new_params_s_prime_list[j] = min(P, current_params_s_tuple[j] + current_plan_A[j])
            
            new_params_s_prime_tuple = tuple(new_params_s_prime_list)
            idx_S_prime = get_idx_from_params(new_params_s_prime_tuple)
            
            # Update the cost to reach S_new
            dp[idx_S_prime] = min(dp[idx_S_prime], dp[current_idx] + current_plan_cost)
            
    # The goal is to reach state (P, P, ..., P)
    target_params_tuple = tuple([P] * K)
    min_total_cost = dp[get_idx_from_params(target_params_tuple)]

    if min_total_cost == math.inf:
        sys.stdout.write("-1
")
    else:
        # Costs are integers, so the result should be an integer.
        sys.stdout.write(str(int(min_total_cost)) + "
")

if __name__ == '__main__':
    main()