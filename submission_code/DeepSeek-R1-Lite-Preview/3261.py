from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        def check(x: int) -> bool:
            n = len(nums)
            groups = 0
            i = 0
            while i < n:
                current_and = nums[i]
                if current_and > x:
                    return False
                j = i + 1
                while j < n:
                    current_and &= nums[j]
                    if current_and > x:
                        break
                    j += 1
                if j > i:
                    groups += 1
                    i = j
                else:
                    return False
            return groups <= n - k
        
        # Find the initial OR of the array
        initial_or = 0
        for num in nums:
            initial_or |= num
        
        # Binary search over possible OR values
        low = 0
        high = initial_or
        while low < high:
            mid = (low + high) // 2
            if check(mid):
                high = mid
            else:
                low = mid + 1
        return low