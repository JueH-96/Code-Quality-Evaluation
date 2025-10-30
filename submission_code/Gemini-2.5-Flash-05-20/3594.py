import collections
import math
from typing import List

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Step 1: Calculate the total sum of all elements in nums.
        # This sum is equal to (sum of n-2 special numbers) * 2 + outlier.
        total_sum = sum(nums)
        
        # Step 2: Create a frequency map (Counter) to efficiently check
        # the existence and count of numbers within nums. This is crucial
        # for handling cases where special numbers, their sum, or the outlier
        # share the same value.
        counts = collections.Counter(nums)
        
        # Step 3: Initialize max_outlier to a very small number.
        # The problem guarantees that at least one potential outlier exists,
        # so this variable will definitely be updated with a valid outlier value.
        max_outlier = -math.inf 
        
        # Step 4: Iterate through each unique number present in nums.
        # We assume each unique number 's_num_candidate' is a potential sum of the
        # n-2 special numbers.
        for s_num_candidate in counts:
            # Based on the derived equation: total_sum = 2 * s_num_candidate + o_num_candidate,
            # we can calculate the value of the corresponding potential outlier.
            o_num_candidate = total_sum - 2 * s_num_candidate
            
            # Step 4b: Check if both 's_num_candidate' (as the sum) and
            # 'o_num_candidate' (as the outlier) can be chosen from 'nums'
            # such that they correspond to distinct indices.
            is_valid_pair = False
            if s_num_candidate == o_num_candidate:
                # If the value for the sum of special numbers is the same as the outlier's value,
                # we need at least two occurrences of this value in 'nums'.
                # One occurrence will be used as the sum, and the other as the outlier.
                if counts.get(s_num_candidate, 0) >= 2:
                    is_valid_pair = True
            else:
                # If the values are different, we simply need at least one occurrence of
                # 's_num_candidate' and at least one occurrence of 'o_num_candidate'.
                # Since their values are distinct, they will naturally occupy distinct indices.
                if counts.get(s_num_candidate, 0) >= 1 and counts.get(o_num_candidate, 0) >= 1:
                    is_valid_pair = True
            
            # Step 4c: If the current pair (s_num_candidate, o_num_candidate) is valid,
            # then o_num_candidate is a potential outlier. We update max_outlier
            # if this o_num_candidate is greater than the current maximum found so far.
            if is_valid_pair:
                max_outlier = max(max_outlier, o_num_candidate)
                
        # Step 5: Return the largest potential outlier found.
        # Since all input numbers are integers, the outlier will also be an integer.
        return int(max_outlier)