class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the dominant element
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        dominant = None
        dominant_count = 0
        for num, freq in count.items():
            if freq * 2 > n:
                dominant = num
                dominant_count = freq
                break
        
        # Try each split position
        left_dominant_count = 0
        for i in range(n - 1):
            if nums[i] == dominant:
                left_dominant_count += 1
            
            left_length = i + 1
            right_length = n - i - 1
            right_dominant_count = dominant_count - left_dominant_count
            
            # Check if dominant element is dominant in both parts
            if (left_dominant_count * 2 > left_length and 
                right_dominant_count * 2 > right_length):
                return i
        
        return -1