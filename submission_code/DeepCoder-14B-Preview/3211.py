class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Binary search on the possible number of groups (k)
        left = 1
        right = n
        best = 0
        
        while left <= right:
            mid = (left + right) // 2
            if self.is_possible(mid, nums):
                best = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return best
    
    def is_possible(self, k: int, nums: List[int]) -> bool:
        if k == 0:
            return False
        n = len(nums)
        prev_sum = 0
        current_sum = 0
        groups = 0
        i = 0
        
        while i < n and groups < k:
            current_sum += nums[i]
            i += 1
            if current_sum >= prev_sum:
                groups += 1
                prev_sum = current_sum
                current_sum = 0
        
        return groups == k and i == n