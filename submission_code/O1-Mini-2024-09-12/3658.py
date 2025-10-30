from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        def is_feasible(D: int) -> bool:
            n = len(nums)
            # Check all consecutive non -1 elements
            for i in range(n - 1):
                if nums[i] != -1 and nums[i + 1] != -1:
                    if abs(nums[i] - nums[i + 1]) > D:
                        return False
            
            # Collect all numbers adjacent to -1
            adjacent_nums = []
            for i in range(n):
                if nums[i] == -1:
                    if i > 0 and nums[i - 1] != -1:
                        adjacent_nums.append(nums[i - 1])
                    if i < n -1 and nums[i +1] != -1:
                        adjacent_nums.append(nums[i +1])
            
            if not adjacent_nums:
                # All elements are -1, can set x = y, thus D can be 0
                return True
            
            # Determine the intersection interval [L, R]
            L = max(num - D for num in adjacent_nums)
            R = min(num + D for num in adjacent_nums)
            
            if L > R:
                return False
            
            # Check if the intersection interval allows x and y with |x - y| <= D
            if R - L <= D:
                return True
            else:
                return False
        
        low = 0
        high = 10**9
        while low < high:
            mid = (low + high) // 2
            if is_feasible(mid):
                high = mid
            else:
                low = mid + 1
        return low