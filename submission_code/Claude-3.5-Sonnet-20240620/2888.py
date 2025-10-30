class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the dominant element and its frequency
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        dominant = max(count, key=count.get)
        total_freq = count[dominant]
        
        # Check for valid splits
        left_freq = 0
        for i in range(n - 1):
            if nums[i] == dominant:
                left_freq += 1
            
            left_length = i + 1
            right_length = n - left_length
            
            if left_freq * 2 > left_length and (total_freq - left_freq) * 2 > right_length:
                return i
        
        return -1