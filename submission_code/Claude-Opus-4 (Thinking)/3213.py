class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # Find the maximum element in the array
        max_element = max(nums)
        
        n = len(nums)
        count = 0
        left = 0
        max_count = 0
        
        # Use sliding window with right pointer
        for right in range(n):
            # If current element is the max element, increment counter
            if nums[right] == max_element:
                max_count += 1
            
            # Shrink window from left while we have at least k occurrences
            while max_count >= k:
                if nums[left] == max_element:
                    max_count -= 1
                left += 1
            
            # All subarrays from indices 0 to (left-1) ending at right
            # contain at least k occurrences of max element
            count += left
        
        return count