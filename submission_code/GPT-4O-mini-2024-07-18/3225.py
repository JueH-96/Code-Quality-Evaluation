class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        left = 0
        count = {}
        max_length = 0
        
        for right in range(len(nums)):
            count[nums[right]] = count.get(nums[right], 0) + 1
            
            while count[nums[right]] > k:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    del count[nums[left]]
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length