# YOUR CODE HERE
import sys

# Set higher recursion depth for deep stacks possibly
# sys.setrecursionlimit(200005) 

def solve():
    N = int(sys.stdin.readline())
    
    # Handle edge cases N=0, N=1
    if N == 0: 
        print(1) # Empty sequence has 1 way (0 operations)
        return
    
    # Read the input array A
    A = list(map(int, sys.stdin.readline().split()))

    if N == 1:
         # 1 element array, initial state may or may not match A[0],
         # but 0 operations are needed. The empty sequence is the only way.
         print(1) 
         return

    MOD = 998244353

    # Precompute factorials and inverse factorials for combinations
    # Factorials up to N are needed for nCr calculations
    fact = [1] * (N + 1)
    inv_fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    
    # Check if N=0 before accessing inv_fact[N] is not needed due to N>=1 check
    inv_fact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N - 1, -1, -1):
         inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

    # Modular combinations function: nCr % MOD
    def nCr_mod(n, r):
        if r < 0 or r > n: return 0
        # Base cases: C(n,0) = 1, C(n,n) = 1
        if r == 0 or r == n: return 1
        # Optimization: C(n, r) = C(n, n-r), use smaller r
        if r > n // 2: r = n - r
        
        # Calculate n! / (r! * (n-r)!) mod MOD
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n - r]) % MOD
        return (num * den) % MOD

    # Stack implementation using a list. Stores indices from A.
    stack_indices = [-1] # Use -1 as dummy bottom index representing state before start
    
    # DP state dictionaries associated with stack pointer index k (position in stack_indices list)
    # dp[k]: # ways ending state corresponding to stack_indices[k]
    # ops_cnt[k]: # total operations count K for state stack_indices[k]
    # choices_sum[k]: # sum C = sum(num_elements_inside // 2) for state stack_indices[k]
    
    dp = {0: 1} # dp[stack_list_idx], base case for dummy index -1 state
    ops_cnt = {0: 0}
    choices_sum = {0: 0}

    # stack_ptr points to the index in stack_indices list for the current top element
    stack_ptr = 0 

    # Iterate through A array elements 0 to N-1
    for i in range(N): 
        
        # Default values to update state with (used if no pop occurs)
        # State of the element below the potential new top 'i'
        current_dp = dp[stack_ptr]
        current_ops_cnt = ops_cnt[stack_ptr]
        current_choices_sum = choices_sum[stack_ptr]

        # Check current element A[i] against stack top A[stack_indices[stack_ptr]]
        # stack_ptr > 0 ensures we don't check against dummy index -1
        if stack_ptr > 0 and A[stack_indices[stack_ptr]] == A[i]:
             
            j_idx = stack_indices[stack_ptr] # The actual index in A array corresponding to stack top

            # Condition for valid operation interval: distance >= 2
            if i >= j_idx + 2: 
                
                # Retrieve state of the element being popped
                last_dp = dp[stack_ptr]
                last_ops_cnt = ops_cnt[stack_ptr]
                last_choices_sum = choices_sum[stack_ptr]
                
                # Pop conceptually by decrementing stack_ptr
                stack_ptr -= 1 
                
                # Update state of the new stack top (element below popped one)
                # This aggregates results from the completed interval (j_idx, i)
                dp[stack_ptr] = (dp[stack_ptr] + last_dp) % MOD
                ops_cnt[stack_ptr] = last_ops_cnt + 1 # One more operation completed
                
                num_elements_inside = i - j_idx - 1 # Number of elements k between j_idx and i
                current_choices = num_elements_inside // 2 # Choices associated with this operation interval
                choices_sum[stack_ptr] = last_choices_sum + current_choices
                
                # After update, prepare state for the upcoming push of 'i'
                # The state for 'i' will be based on the UPDATED state of the new stack top
                current_dp = dp[stack_ptr]
                current_ops_cnt = ops_cnt[stack_ptr]
                current_choices_sum = choices_sum[stack_ptr]

            # else: # Adjacent case: i == j_idx + 1. 
                 # The elements A[j_idx] and A[i] are adjacent and identical.
                 # No pop occurs. No state update needed here for the element j_idx.
                 # The state for 'i' will be based on the state of 'j_idx' (current top).
                 # This is covered by the default initial values assigned to current_dp etc.
                 pass # Explicitly do nothing for this case, logic flows to push 'i'

        # If mismatch OR stack top is dummy OR adjacent match case happened:
        # Push i onto stack
        stack_ptr += 1
        
        # Manage the stack_indices list size dynamically or reuse slots
        if stack_ptr == len(stack_indices):
            stack_indices.append(i)
        else:
            # Reuse slot if stack had shrunk previously
            stack_indices[stack_ptr] = i
        
        # Initialize/Set state for the new stack top `i` (at `stack_ptr`)
        # Uses the values prepared in `current_dp`, `current_ops_cnt`, `current_choices_sum`
        # which correctly reflect the state of the element below the new top.
        dp[stack_ptr] = current_dp
        ops_cnt[stack_ptr] = current_ops_cnt
        choices_sum[stack_ptr] = current_choices_sum

    # Final calculation based on the final stack top state after processing all N elements
    final_total_ops = ops_cnt[stack_ptr] # Total operations K
    final_total_choices = choices_sum[stack_ptr] # Sum of choices C = Sum (k_i // 2)
    
    # The final combinatorial factor is nCr(K + C, C) where K=total ops, C=sum of choices
    comb_factor = nCr_mod(final_total_ops + final_total_choices, final_total_choices)
    
    # Final answer is dp value of final stack top times the combinatorial factor
    final_ans = (dp[stack_ptr] * comb_factor) % MOD
    print(final_ans)

solve()