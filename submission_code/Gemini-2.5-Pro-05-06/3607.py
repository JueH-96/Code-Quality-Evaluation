import math

class Solution:
  def minOperations(self, nums: list[int]) -> int:
    
    MAX_VAL = 0
    if not nums:
        # Based on constraints (1 <= nums.length), nums is never empty.
        # If it could be, this would be a valid return.
        return 0 
        
    for x in nums:
        MAX_VAL = max(MAX_VAL, x)

    if MAX_VAL == 1:
        # If all numbers are 1 (since nums[i] >= 1), 0 operations needed.
        return 0

    # Sieve of Eratosthenes to find Smallest Prime Factor (SPF)
    spf = [0] * (MAX_VAL + 1)
    if MAX_VAL >= 1: # Should always be true if MAX_VAL > 1
        spf[1] = 1 # 1 is not prime, handle as special case. SPF is nominally undefined.
    
    # Correct SPF Sieve:
    # Initialize spf[i] = i for all i
    # Then iterate primes p and update spf[p*k] = p
    # This simpler version also works:
    for i in range(2, MAX_VAL + 1):
        if spf[i] == 0: # i is prime, its SPF is i itself
            for j in range(i, MAX_VAL + 1, i): # Iterate through multiples of i
                if spf[j] == 0: # If spf[j] is not set yet by an even smaller prime
                    spf[j] = i
    
    memo_reachable_states = {}
    def get_reachable_states(num_val):
        if num_val in memo_reachable_states:
            return memo_reachable_states[num_val]

        states = [{'val': num_val, 'cost': 0}]
        
        # is_num_prime: checks if num_val > 1 and its smallest prime factor is itself.
        is_num_prime = (num_val > 1 and spf[num_val] == num_val)
        
        # If num_val is composite (i.e., > 1 and not prime)
        if num_val > 1 and not is_num_prime:
            # Composite num_val becomes spf[num_val] in 1 op.
            # spf[num_val] is prime, so no further changes from that state.
            # Since spf[num_val] < num_val for composite numbers,
            # this list will be sorted if we append then sort.
            states.append({'val': spf[num_val], 'cost': 1})
            states.sort(key=lambda s: s['val']) 
        
        memo_reachable_states[num_val] = states
        return states

    # Dynamic Programming
    current_dp_list = []
    
    # Base case for nums[0]
    initial_states_n0 = get_reachable_states(nums[0])
    for s in initial_states_n0:
        current_dp_list.append({'val': s['val'], 'ops': s['cost']})

    # Loop for nums[1] through nums[n-1]
    for i in range(1, len(nums)):
        prev_dp_list = current_dp_list 
        current_dp_list = []          
        
        reachable_states_for_current_num = get_reachable_states(nums[i])
        
        ptr_prev = 0 
        min_ops_from_prev_prefix = float('inf')
        
        for s_curr in reachable_states_for_current_num: 
            val_curr = s_curr['val']   
            cost_curr = s_curr['cost'] 
            
            while ptr_prev < len(prev_dp_list) and prev_dp_list[ptr_prev]['val'] <= val_curr:
                min_ops_from_prev_prefix = min(min_ops_from_prev_prefix, prev_dp_list[ptr_prev]['ops'])
                ptr_prev += 1
            
            if min_ops_from_prev_prefix != float('inf'):
                current_dp_list.append({'val': val_curr, 
                                        'ops': cost_curr + min_ops_from_prev_prefix})
        
        if not current_dp_list:
            return -1
            
    if not current_dp_list: # Should be caught by the check inside loop
        return -1
        
    final_min_ops = float('inf')
    for s in current_dp_list:
        final_min_ops = min(final_min_ops, s['ops'])
            
    return final_min_ops if final_min_ops != float('inf') else -1