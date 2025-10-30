from typing import List

class Solution:
  def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
    n = len(nums)
    
    # dp[j][l] will store the maximum reduction achievable using j op1s and l op2s
    # Initialize with -1 to signify states not yet reached or impossible.
    # Reductions are non-negative, so -1 works as a sentinel for "unreachable".
    # dp[0][0] = 0 means 0 reduction with 0 ops of each type.
    dp = [[-1 for _ in range(op2 + 1)] for _ in range(op1 + 1)]
    dp[0][0] = 0
    
    initial_sum = 0
    for x_val in nums:
        initial_sum += x_val
        
    if initial_sum == 0: # Optimization: if sum is 0, no reduction possible/needed
        return 0

    for x_val_in_nums in nums: # Iterate through each number in nums
        x = x_val_in_nums
        
        if x == 0: # Optimization: 0 cannot be reduced further by these operations
            continue

        # Calculate potential reductions for current number x
        # B1: Reduction from Op1 only
        # ceil(x/2) is (x + 1) // 2 for non-negative x
        B1 = x - (x + 1) // 2
        
        # B2: Reduction from Op2 only
        B2_val = k # Reduction amount is k
        B2_possible = (x >= k) # Op2 is only possible if x >= k
        
        # B3: Reduction from applying both Op1 and Op2 to x.
        # We need to find the better order: (Op1 then Op2) or (Op2 then Op1).
        
        # Path A: Op1 then Op2
        val_after_op1 = (x + 1) // 2
        reduction_from_op1_first = x - val_after_op1 # Reduction due to Op1
        
        B3_A = reduction_from_op1_first
        # Then apply Op2 to val_after_op1
        if val_after_op1 >= k: # Op2 is applicable
            B3_A += k # Additional reduction k from Op2
        
        B3_val = B3_A # Initialize B3_val with Path A result
        
        # Path B: Op2 then Op1
        if x >= k: # Original Op2 application condition
            val_after_op2 = x - k
            reduction_from_op2_first = k # Reduction due to Op2
            
            # Then apply Op1 to val_after_op2
            reduction_from_op1_second = val_after_op2 - (val_after_op2 + 1) // 2
            B3_B = reduction_from_op2_first + reduction_from_op1_second
            
            if B3_B > B3_val: # If Path B gives more reduction
                 B3_val = B3_B
        # If x < k, Path B (starting with Op2) is not possible. B3_val effectively remains B3_A.

        # Iterate dp table backwards. This ensures that when we calculate dp[j_ops_count][l_ops_count]
        # using states like dp[j_ops_count-1][l_ops_count], these states
        # refer to the values *before* considering the current element x.
        for j_ops_count in range(op1, -1, -1): # Number of op1s used
            for l_ops_count in range(op2, -1, -1): # Number of op2s used
                
                # Base case for dp[j_ops_count][l_ops_count]: its value from the previous iteration 
                # (i.e., after processing the previous number in nums, or initial if this is the first num).
                # This represents not using any operation on the current 'x'.
                # All updates below try to achieve a better reduction by using an operation on 'x'.

                # Option to apply T3 (Both Ops) to x. Consumes 1 op1 and 1 op2.
                if j_ops_count >= 1 and l_ops_count >= 1 and dp[j_ops_count-1][l_ops_count-1] != -1:
                    reduction_if_T3 = dp[j_ops_count-1][l_ops_count-1] + B3_val
                    if reduction_if_T3 > dp[j_ops_count][l_ops_count]:
                        dp[j_ops_count][l_ops_count] = reduction_if_T3
                                   
                # Option to apply T1 (Op1 only) to x. Consumes 1 op1.
                if j_ops_count >= 1 and dp[j_ops_count-1][l_ops_count] != -1:
                    reduction_if_T1 = dp[j_ops_count-1][l_ops_count] + B1
                    if reduction_if_T1 > dp[j_ops_count][l_ops_count]:
                         dp[j_ops_count][l_ops_count] = reduction_if_T1
                
                # Option to apply T2 (Op2 only) to x. Consumes 1 op2.
                if l_ops_count >= 1 and B2_possible and dp[j_ops_count][l_ops_count-1] != -1:
                    reduction_if_T2 = dp[j_ops_count][l_ops_count-1] + B2_val
                    if reduction_if_T2 > dp[j_ops_count][l_ops_count]:
                        dp[j_ops_count][l_ops_count] = reduction_if_T2
        # The order of T1, T2, T3 updates for dp[j_ops_count][l_ops_count] does not matter critically here
        # because the states dp[...-1][...-1] are guaranteed to be from the previous outer loop iteration
        # (processing previous number) due to the backward loops for j_ops_count and l_ops_count.

    max_total_reduction = 0
    for j_ops_final in range(op1 + 1):
        for l_ops_final in range(op2 + 1):
            # dp state could be -1 if never reached. Max reduction must be non-negative.
            if dp[j_ops_final][l_ops_final] > max_total_reduction:
                max_total_reduction = dp[j_ops_final][l_ops_final]
            
    return initial_sum - max_total_reduction