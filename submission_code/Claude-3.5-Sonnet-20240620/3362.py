class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total = n * (n + 1) // 2
        mid = (total + 1) // 2
        
        left, right = 1, n
        while left < right:
            mid_val = (left + right) // 2
            count = self.countLessEqual(nums, mid_val)
            if count < mid:
                left = mid_val + 1
            else:
                right = mid_val
        
        return left
    
    def countLessEqual(self, nums: List[int], target: int) -> int:
        count = 0
        left = 0
        unique = set()
        
        for right in range(len(nums)):
            unique.add(nums[right])
            while len(unique) > target:
                unique.remove(nums[left])
                left += 1
            count += right - left + 1
        
        return count