from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If any number is strictly less than k, it's impossible (we can only decrease values)
        if any(num < k for num in nums):
            return -1
        
        # Gather all distinct values that are >= k
        distinct_vals = sorted(set(num for num in nums if num >= k), reverse=True)
        
        # Ensure k is in the list of distinct values, so we eventually reduce to k
        if k not in distinct_vals:
            distinct_vals.append(k)
        
        # The number of operations needed is the count of these distinct values minus 1
        return len(distinct_vals) - 1