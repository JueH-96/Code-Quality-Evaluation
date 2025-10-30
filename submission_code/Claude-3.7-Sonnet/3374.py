class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            count += 1  # Count the single-element subarray
            for j in range(i+1, n):
                if nums[j] == nums[j-1]:
                    break
                count += 1
        
        return count