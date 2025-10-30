class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Find the dominant element of the entire array
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        n = len(nums)
        dominant = max(count, key=count.get)
        dominant_count = count[dominant]
        
        # Check each possible split
        current_dominant_count = 0
        for i in range(n - 1):  # We need at least one element in the right subarray
            if nums[i] == dominant:
                current_dominant_count += 1
            
            # Check if the dominant element is the same in both subarrays
            left_size = i + 1
            right_size = n - left_size
            
            if current_dominant_count * 2 > left_size and (dominant_count - current_dominant_count) * 2 > right_size:
                return i
        
        return -1