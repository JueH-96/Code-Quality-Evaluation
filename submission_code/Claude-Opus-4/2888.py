class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Find the dominant element in the entire array
        from collections import Counter
        n = len(nums)
        count = Counter(nums)
        
        dominant = None
        for num, freq in count.items():
            if freq * 2 > n:
                dominant = num
                break
        
        # Track the frequency of the dominant element as we iterate
        left_count = 0
        total_count = count[dominant]
        
        # Check each possible split point
        for i in range(n - 1):
            if nums[i] == dominant:
                left_count += 1
            
            left_length = i + 1
            right_length = n - i - 1
            right_count = total_count - left_count
            
            # Check if dominant element is dominant in both parts
            if (left_count * 2 > left_length and 
                right_count * 2 > right_length):
                return i
        
        return -1