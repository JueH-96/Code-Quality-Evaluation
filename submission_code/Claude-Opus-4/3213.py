class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_element = max(nums)
        n = len(nums)
        count = 0
        left = 0
        max_count = 0
        
        # Use sliding window
        for right in range(n):
            # If current element is the max element, increment count
            if nums[right] == max_element:
                max_count += 1
            
            # Shrink window from left while we have at least k max elements
            while max_count >= k:
                # All subarrays from current left to any position >= right are valid
                count += n - right
                
                # Move left pointer and update count if needed
                if nums[left] == max_element:
                    max_count -= 1
                left += 1
        
        return count