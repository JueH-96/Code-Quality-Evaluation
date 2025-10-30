from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        """
        Finds the maximum frequency of the value k after performing one operation.
        The operation consists of selecting a subarray and adding an integer x to all its elements.
        """
        
        # The final frequency of k is the initial frequency plus a "gain" from one operation.
        # An operation involves picking a subarray `nums[i..j]` and a value `v` to convert to `k`.
        # This requires adding `x = k - v` to all elements in the subarray.
        # The gain for this operation is `count(v in nums[i..j]) - count(k in nums[i..j])`.
        # Our goal is to maximize this gain over all possible `v` and all subarrays `[i..j]`.
        
        # For a fixed `v`, maximizing the gain is a Maximum Subarray Sum problem.
        # We can solve it efficiently with Kadane's algorithm.

        initial_k_count = nums.count(k)
        max_gain = 0

        # Iterate through all possible original values 'v' that can be converted to 'k'.
        # The problem constraints state values are between 1 and 50.
        for v in range(1, 51):
            if v == k:
                continue

            # Use Kadane's algorithm to find the maximum gain for converting 'v' to 'k'.
            # We can think of a conceptual array where:
            # - elements equal to 'v' have a score of +1
            # - elements equal to 'k' have a score of -1 (as they are lost)
            # - all other elements have a score of 0
            current_gain = 0
            max_gain_for_v = 0
            for num in nums:
                if num == v:
                    current_gain += 1
                elif num == k:
                    current_gain -= 1
                
                # If the running gain becomes negative, it's better to start a new subarray.
                if current_gain < 0:
                    current_gain = 0
                
                max_gain_for_v = max(max_gain_for_v, current_gain)
            
            max_gain = max(max_gain, max_gain_for_v)
            
        return initial_k_count + max_gain