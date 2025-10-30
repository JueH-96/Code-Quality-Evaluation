class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        max_val = max(nums)
        n = len(nums)
        
        for i in range(n):
            if nums[i] == max_val:
                j = i
                while j < n and nums[j] <= max_val:
                    j += 1
                count += (j - i) * (j - i + 1) // 2
                i = j - 1
        
        return count