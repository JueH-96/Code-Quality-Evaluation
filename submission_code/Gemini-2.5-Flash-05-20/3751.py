import collections
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Step 1: Calculate the initial frequency of k in the array.
        # This serves as a baseline, covering the case where the optimal operation
        # is to effectively do nothing (i.e., choose x=0).
        initial_k_freq = 0
        for num in nums:
            if num == k:
                initial_k_freq += 1
        
        # Initialize the maximum overall frequency found so far with the baseline.
        max_overall_freq = initial_k_freq
        
        # Step 2: Iterate through all possible values 'V' that elements in nums[i..j] could originally have
        # and be transformed into 'k'.
        # Constraints state 1 <= nums[i] <= 50, so V can be any integer in this range.
        for V in range(1, 51):
            # If V is k, then x = k - V = 0. Adding 0 to a subarray doesn't change it.
            # In this case, the frequency of k is just its initial frequency.
            # This case is already covered by 'initial_k_freq', so we skip it to avoid
            # redundant calculations and potential misinterpretations of the scoring
            # system designed for V != k.
            if V == k:
                continue 
            
            # For V != k, the value 'x' to be added to the subarray is x = k - V.
            # If an element 'nums[p]' is inside nums[i..j]:
            #   - If nums[p] == V, it becomes V + (k - V) = k (contributes +1 to new k's).
            #   - If nums[p] == k, it becomes k + (k - V) (which is not k since V != k, so it removes a k, contributing -1).
            #   - Otherwise, it does not become k (contributes 0).
            # The total frequency of k will be:
            # (initial_k_freq) + (count of V's in [i..j] - count of k's in [i..j]).
            # We want to maximize the (count of V's - count of k's) part using Kadane's algorithm.
            
            current_max_subarray_score = 0  # Stores the maximum (count of V's - count of k's) for any subarray
            current_subarray_sum = 0        # Stores the current sum for the Kadane's window
            
            for num_val in nums:
                score_for_num = 0
                if num_val == V:
                    score_for_num = 1      # V values are good, contribute +1
                elif num_val == k:
                    score_for_num = -1     # K values inside the subarray are bad, contribute -1
                # Other values (not V or k) contribute 0 to the sum difference
                
                current_subarray_sum += score_for_num
                
                # Kadane's algorithm: If the current sum goes negative, it's better to start a new subarray
                # from the next element, as a negative prefix will only reduce future sums.
                # Resetting to 0 implicitly allows for an empty subarray (which has a score of 0).
                if current_subarray_sum < 0:
                    current_subarray_sum = 0
                
                # Update the maximum subarray sum found so far for the current V.
                current_max_subarray_score = max(current_max_subarray_score, current_subarray_sum)
            
            # The maximum frequency for this specific choice of 'V' is the initial k frequency
            # plus the best possible increase (or smallest decrease) achieved by applying the operation.
            max_overall_freq = max(max_overall_freq, initial_k_freq + current_max_subarray_score)
            
        return max_overall_freq