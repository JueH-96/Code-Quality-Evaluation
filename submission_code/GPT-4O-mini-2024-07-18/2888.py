class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import Counter
        
        n = len(nums)
        freq = Counter(nums)
        dominant_element = max(freq, key=freq.get)
        dominant_freq = freq[dominant_element]
        
        # Check if the dominant element is indeed dominant
        if dominant_freq * 2 <= n:
            return -1
        
        left_count = 0
        
        for i in range(n - 1):
            if nums[i] == dominant_element:
                left_count += 1
            
            right_count = dominant_freq - left_count
            
            if left_count * 2 > (i + 1) and right_count * 2 > (n - (i + 1)):
                return i
        
        return -1