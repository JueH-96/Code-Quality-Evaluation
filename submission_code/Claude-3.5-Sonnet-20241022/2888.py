class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Find the dominant element in the full array
        n = len(nums)
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        dominant = max(count.items(), key=lambda x: x[1])[0]
        total_freq = count[dominant]
        
        # Track frequency of dominant element in left part
        left_freq = 0
        
        for i in range(n-1):
            if nums[i] == dominant:
                left_freq += 1
            
            # Check if dominant element is still dominant in both parts
            left_len = i + 1
            right_len = n - left_len
            right_freq = total_freq - left_freq
            
            if left_freq * 2 > left_len and right_freq * 2 > right_len:
                return i
                
        return -1