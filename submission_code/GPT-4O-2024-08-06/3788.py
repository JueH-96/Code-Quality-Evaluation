class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Use a sliding window approach with a set to track unique elements
        max_sum = 0
        current_sum = 0
        left = 0
        seen = set()
        
        for right in range(len(nums)):
            while nums[right] in seen:
                # Remove the leftmost element until nums[right] can be added
                seen.remove(nums[left])
                current_sum -= nums[left]
                left += 1
            
            # Add the current element to the set and update the current sum
            seen.add(nums[right])
            current_sum += nums[right]
            
            # Update the maximum sum found so far
            max_sum = max(max_sum, current_sum)
        
        return max_sum