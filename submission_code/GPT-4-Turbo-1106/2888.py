class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each element in nums
        freq = Counter(nums)
        
        # Find the dominant element and its frequency
        dominant_element = max(freq, key=freq.get)
        dominant_count = freq[dominant_element]
        
        # Initialize counters for the dominant element in each subarray
        left_dominant_count = 0
        right_dominant_count = dominant_count
        
        # Iterate through the array to find the minimum index for a valid split
        for i in range(len(nums) - 1):
            if nums[i] == dominant_element:
                left_dominant_count += 1
                right_dominant_count -= 1
            
            # Check if both subarrays have the same dominant element
            if left_dominant_count * 2 > i + 1 and right_dominant_count * 2 > len(nums) - i - 1:
                return i + 1
        
        # If no valid split is found, return -1
        return -1