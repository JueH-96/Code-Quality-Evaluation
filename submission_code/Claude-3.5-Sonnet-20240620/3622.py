class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        n = len(nums)
        nums.sort()
        
        def can_achieve_frequency(freq):
            if freq > n:
                return False
            
            for i in range(n - freq + 1):
                target = nums[i + freq - 1]
                operations_needed = sum(target - num for num in nums[i:i+freq])
                if operations_needed <= k * numOperations:
                    return True
            
            return False
        
        left, right = 1, n + 1
        while left < right:
            mid = (left + right) // 2
            if can_achieve_frequency(mid):
                left = mid + 1
            else:
                right = mid
        
        return left - 1