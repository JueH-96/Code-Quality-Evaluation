class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # Iterate k, the number of operations, from 1 upwards.
        # A maximum iteration of k up to around 60-70 is sufficient.
        # This is because the maximum popcount for numbers that can be formed
        # by num1 - k*num2 within the given constraints and k in this range
        # is typically much smaller than 60-70. If k exceeds that max popcount,
        # the popcount(target_val) <= k condition will almost certainly be met,
        # leaving only target_val >= k to be satisfied.
        # Using 65 as an upper bound for k (meaning k from 1 to 64) is robust.
        for k in range(1, 65): 
            # Calculate the value that needs to be formed by k powers of 2.
            target_val = num1 - k * num2
            
            # Condition 1: target_val must be at least k.
            # This ensures that target_val is positive and large enough
            # to be represented by k terms (each term being at least 2^0=1).
            if target_val < k:
                # If num2 is non-negative, (1 + num2) is non-negative.
                # The expression k * (1 + num2) is non-decreasing with k.
                # So if num1 < k * (1 + num2) (which means target_val < k)
                # for the current k, it will also be true for all larger k.
                # Thus, no solution exists for this or any larger k, so we can break.
                # If num2 is negative, (1 + num2) is negative or zero.
                # In these cases, num1 >= k * (1 + num2) (equivalent to target_val >= k)
                # is always true for num1 >= 1 and k >= 1.
                # So, this 'if' block will only be entered if num2 >= 0 and no solution exists.
                if num2 >= 0:
                    return -1 # No solution possible
                # If num2 < 0, target_val >= k always holds, so this 'if' condition is not met.
                # We simply continue to the next iteration if the logic were to somehow land here.
                # (But it won't due to the analysis above).
            
            # Condition 2: popcount(target_val) must be less than or equal to k.
            # popcount is the minimum number of powers of 2 (distinct) needed to form target_val.
            # If we need more terms than popcount, we can "split" 2^j into 2^(j-1) + 2^(j-1).
            pop_count = bin(target_val).count('1')
            
            if pop_count <= k:
                # If both conditions are met, this is the minimum k because we iterate k in increasing order.
                return k
        
        # If the loop finishes without finding a valid k, it's impossible.
        return -1