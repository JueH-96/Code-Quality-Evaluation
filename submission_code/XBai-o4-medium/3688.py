from typing import List
from collections import defaultdict

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            current_sum = 0
            max_sum = float('-inf')
            for num in arr:
                current_sum = max(num, current_sum + num)
                max_sum = max(max_sum, current_sum)
            return max_sum
        
        original_max = kadane(nums)
        
        # Precompute frequency of each number
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # Find all unique negative numbers
        unique_negatives = set()
        for num in nums:
            if num < 0:
                unique_negatives.add(num)
        
        max_result = original_max
        
        for x in unique_negatives:
            # Check if after removing x, the array is not empty
            if freq[x] == len(nums):
                continue  # cannot remove x
            # Compute the max subarray sum after removing x
            current_sum = 0
            current_max = float('-inf')
            for num in nums:
                if num == x:
                    continue
                current_sum = max(num, current_sum + num)
                current_max = max(current_max, current_sum)
            if current_max > max_result:
                max_result = current_max
        
        return max_result