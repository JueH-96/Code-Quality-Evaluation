from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Compute the initial OR of all elements
        initial_or = 0
        for num in nums:
            initial_or |= num
        
        low = 0
        high = initial_or
        answer = initial_or
        
        def is_possible(mid):
            groups = 0
            current_and = None
            for num in nums:
                if current_and is None:
                    current_and = num
                else:
                    current_and &= num
                # Check if current_and is a subset of mid's bits
                if (current_and & mid) == current_and:
                    groups += 1
                    current_and = None
            return groups >= (n - k)
        
        while low <= high:
            mid_val = (low + high) // 2
            if is_possible(mid_val):
                answer = mid_val
                high = mid_val - 1
            else:
                low = mid_val + 1
        return answer