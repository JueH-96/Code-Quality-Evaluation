class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        left = 0
        freq = {}
        max_length = 0
        
        for right in range(len(nums)):
            num = nums[right]
            freq[num] = freq.get(num, 0) + 1
            
            while freq[num] > k:
                freq[nums[left]] -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length