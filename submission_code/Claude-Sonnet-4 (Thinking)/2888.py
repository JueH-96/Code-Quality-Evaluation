class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import Counter
        
        n = len(nums)
        
        # Find the dominant element
        count = Counter(nums)
        dominant = None
        for num, freq in count.items():
            if freq * 2 > n:
                dominant = num
                break
        
        total_count = count[dominant]
        
        # Try each split index
        left_count = 0
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
            
            right_count = total_count - left_count
            left_length = i + 1
            right_length = n - i - 1
            
            # Check if both parts have the same dominant element
            if left_count * 2 > left_length and right_count * 2 > right_length:
                return i
        
        return -1