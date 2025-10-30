class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = {}
        max_len = 0
        left = 0
        
        for right in range(n):
            count[nums[right]] = count.get(nums[right], 0) + 1
            max_len = max(max_len, count[nums[right]])
            
            if right - left + 1 - max_len > k:
                count[nums[left]] -= 1
                left += 1
        
        return max_len