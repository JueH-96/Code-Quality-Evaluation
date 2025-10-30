class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import Counter
        
        n = len(nums)
        
        # Find the dominant element of the entire array
        counter = Counter(nums)
        dominant_elem = None
        for num, freq in counter.items():
            if freq * 2 > n:
                dominant_elem = num
                break
        
        # Count occurrences of dominant element from left
        left_count = 0
        
        # Try each split position
        for i in range(n - 1):
            # Update left count
            if nums[i] == dominant_elem:
                left_count += 1
            
            # Check if left subarray has dominant element as dominant
            left_length = i + 1
            if left_count * 2 <= left_length:
                continue
            
            # Check if right subarray has dominant element as dominant
            right_count = counter[dominant_elem] - left_count
            right_length = n - left_length
            if right_count * 2 <= right_length:
                continue
            
            # Both subarrays have the same dominant element
            return i
        
        return -1