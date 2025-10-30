class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        # Initialize the count of alternating subarrays
        count = 1
        # Initialize the length of the current alternating subarray
        curr_len = 1
        # Iterate over the array from the second element
        for i in range(1, len(nums)):
            # If the current element is different from the previous one, increment the current length
            if nums[i] != nums[i-1]:
                curr_len += 1
            # If the current element is the same as the previous one, reset the current length and increment the count
            else:
                curr_len = 1
            # Add the current length to the count
            count += curr_len
        # Return the count of alternating subarrays
        return count