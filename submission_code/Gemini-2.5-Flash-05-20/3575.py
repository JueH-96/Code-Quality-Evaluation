import collections
from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Max possible OR value is 2^7 - 1 = 127.
        # MAX_OR_VAL for array indexing 0-127 is 128.
        MAX_OR_VAL = 128 

        # dp_left_at_idx[i] stores a (k+1) x MAX_OR_VAL boolean table.
        # table[c][val] is True if 'val' can be formed by ORing 'c' elements
        # chosen from nums[0...i].
        # Each element in dp_left_at_idx will be a list of lists representing the table.
        dp_left_at_idx = []
        
        # current_table represents the possible OR values using elements up to the previous index (i-1).
        # It's a (k+1) x MAX_OR_VAL boolean table.
        current_table = [[False] * MAX_OR_VAL for _ in range(k + 1)]
        current_table[0][0] = True # Base case: 0 elements OR to 0 is always possible
        
        for num in nums:
            # Create a new_table for the current index i.
            # This table will store results considering elements up to the current 'num'.
            new_table = [[False] * MAX_OR_VAL for _ in range(k + 1)]
            new_table[0][0] = True # 0 elements OR to 0 is always possible
            
            # Case 1: 'num' (nums[i]) is NOT included in the 'c' elements.
            # In this case, the possible OR sums for 'c' elements are simply copied
            # from the 'current_table' (which represents nums[0...i-1]).
            for c in range(k + 1):
                for val in range(MAX_OR_VAL):
                    new_table[c][val] = current_table[c][val]
            
            # Case 2: 'num' (nums[i]) IS included in the 'c' elements.
            # To form 'val' with 'c' elements, we pick 'num' and 'c-1' elements
            # from nums[0...i-1]. Their OR sum 'prev_or' combines with 'num'.
            for c in range(1, k + 1): # Iterate 'c' from 1 up to k
                for prev_or in range(MAX_OR_VAL): # Iterate through possible previous OR sums
                    if current_table[c-1][prev_or]: # If prev_or is achievable with c-1 elements
                        new_table[c][prev_or | num] = True # Then prev_or | num is achievable with c elements
            
            dp_left_at_idx.append(new_table) # Store the table for the current index
            current_table = new_table # Update current_table for the next iteration

        # dp_right_at_idx[i] stores a (k+1) x MAX_OR_VAL boolean table.
        # table[c][val] is True if 'val' can be formed by ORing 'c' elements
        # chosen from nums[i...n-1].
        # Initialize with None, will fill from right to left to maintain original indexing.
        dp_right_at_idx = [None] * n 
        
        # current_table for dp_right_at_idx represents possible OR values
        # from elements starting from the next index (i+1).
        current_table = [[False] * MAX_OR_VAL for _ in range(k + 1)]
        current_table[0][0] = True # Base case: 0 elements OR to 0
        
        # Iterate from n-1 down to 0
        for i in range(n - 1, -1, -1): 
            num = nums[i]
            # Create a new_table for the current index i.
            # This table will store results considering elements from current 'num' to n-1.
            new_table = [[False] * MAX_OR_VAL for _ in range(k + 1)]
            new_table[0][0] = True # 0 elements OR to 0 is always possible
            
            # Case 1: 'num' (nums[i]) is NOT included.
            # Copy from current_table (which represents nums[i+1...n-1]).
            for c in range(k + 1):
                for val in range(MAX_OR_VAL):
                    new_table[c][val] = current_table[c][val]
            
            # Case 2: 'num' (nums[i]) IS included.
            # Pick 'num' and 'c-1' elements from nums[i+1...n-1].
            for c in range(1, k + 1):
                for prev_or in range(MAX_OR_VAL):
                    if current_table[c-1][prev_or]:
                        new_table[c][prev_or | num] = True
            
            dp_right_at_idx[i] = new_table # Store the table for the current index
            current_table = new_table # Update current_table for the next iteration

        max_xor_value = 0

        # Iterate over possible split points 'p'.
        # 'p' is the index of the last element considered for the first 'k' elements (OR1).
        # - The k elements for OR1 are chosen from nums[0...p].
        # - The k elements for OR2 are chosen from nums[p+1...n-1].
        #
        # Constraints on 'p':
        # 1. To pick k elements from nums[0...p], we need p+1 >= k  =>  p >= k-1.
        # 2. To pick k elements from nums[p+1...n-1], we need n-(p+1) >= k => p+1 <= n-k => p <= n-k-1.
        for p in range(k - 1, n - k):
            or1_possibilities = dp_left_at_idx[p][k]       # Get possible OR sums for k elements from nums[0...p]
            or2_possibilities = dp_right_at_idx[p+1][k]     # Get possible OR sums for k elements from nums[p+1...n-1]
            
            # Iterate through all possible OR1 values and OR2 values
            for or1_val in range(MAX_OR_VAL):
                if or1_possibilities[or1_val]:
                    for or2_val in range(MAX_OR_VAL):
                        if or2_possibilities[or2_val]:
                            max_xor_value = max(max_xor_value, or1_val ^ or2_val)
                            
        return max_xor_value