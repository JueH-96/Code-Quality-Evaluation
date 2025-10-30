class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        current_and = (1 << 30) - 1  # Initialize with all bits set to 1
        left = 0

        for right in range(n):
            current_and &= nums[right]
            
            while left <= right and current_and < k:
                current_and = (1 << 30) - 1
                for i in range(left + 1, right + 1):
                    current_and &= nums[i]
                left += 1
            
            if current_and == k:
                count += left + 1

        return count