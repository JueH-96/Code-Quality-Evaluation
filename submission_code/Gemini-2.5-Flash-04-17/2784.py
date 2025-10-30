from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Sort the array in non-decreasing order
        nums.sort()
        
        n = len(nums)
        total_power = 0
        
        # min_sum_up_to_prev stores MinSumNonEmpty(i-1) when processing index i
        # MinSumNonEmpty(k) = Sum over j from 0 to k (nums[j] * 2^(k-j))
        # This represents the sum of the minimums of all non-empty subsets formed using elements from nums[0] to nums[k].
        # Base case: MinSumNonEmpty(-1) = 0 (sum over empty set of subsets).
        min_sum_up_to_prev = 0 
        
        for i in range(n):
            current_num = nums[i]
            
            # We consider the contribution of nums[i] being the maximum element in a subset.
            # A subset S from nums[0...i] where max(S) = current_num must include current_num.
            # The other elements in S must be chosen from nums[0...i-1]. Let this subset be S'.
            # S = {current_num} U S', where S' is a subset of nums[0...i-1].
            # The power of such a subset S is max(S)^2 * min(S) = current_num^2 * min(S).
            # We need to sum current_num^2 * min(S) over all possible subsets S' of nums[0...i-1].
            # Sum(min(S)) over all subsets S' of nums[0...i-1] (min({current_num} U S')):
            # - If S' is empty (1 case): min({current_num}) = current_num. Contribution = current_num^2 * current_num.
            # - If S' is non-empty: min({current_num} U S') = min(S') because current_num >= any element in nums[0...i-1].
            #   The sum over non-empty S' is Sum over non-empty S' of nums[0...i-1] (min(S')).
            #   This is exactly MinSumNonEmpty(i-1), which is stored in min_sum_up_to_prev.
            # The total sum of minimums when current_num is the maximum (in subsets of nums[0...i]) is:
            # current_num (from empty S') + MinSumNonEmpty(i-1) (from non-empty S').
            
            # Calculate current_num^2 modulo MOD
            current_num_sq = (current_num * current_num) % MOD
            
            # Calculate the sum of minimums for subsets where current_num is the maximum
            # This is (current_num + MinSumNonEmpty(i-1)) % MOD
            sum_of_minimums = (current_num + min_sum_up_to_prev) % MOD
            
            # Contribution of current_num as the maximum element from subsets of nums[0...i]
            # This is current_num^2 * (current_num + MinSumNonEmpty(i-1)) % MOD
            contribution = (current_num_sq * sum_of_minimums) % MOD
            
            # Add this contribution to the total power, applying modulo
            total_power = (total_power + contribution) % MOD
            
            # Update min_sum_up_to_prev for the next iteration (when processing nums[i+1]).
            # The new min_sum_up_to_prev will be MinSumNonEmpty(i).
            # We use the recurrence: MinSumNonEmpty(k) = nums[k] + 2 * MinSumNonEmpty(k-1)
            # So, MinSumNonEmpty(i) = nums[i] + 2 * MinSumNonEmpty(i-1).
            # MinSumNonEmpty(i-1) is the current min_sum_up_to_prev.
            min_sum_up_to_current = (current_num + 2 * min_sum_up_to_prev) % MOD
            min_sum_up_to_prev = min_sum_up_to_current
        
        return total_power