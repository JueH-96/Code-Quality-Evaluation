class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        total_alternating_subarrays = 0
        current_length = 1
        
        # Iterate through the array to find alternating subarrays
        for i in range(1, n):
            if nums[i] != nums[i - 1]:
                current_length += 1
            else:
                # Calculate the number of alternating subarrays in the current segment
                total_alternating_subarrays += (current_length * (current_length + 1)) // 2
                current_length = 1
        
        # Add the last segment's alternating subarrays
        total_alternating_subarrays += (current_length * (current_length + 1)) // 2
        
        return total_alternating_subarrays