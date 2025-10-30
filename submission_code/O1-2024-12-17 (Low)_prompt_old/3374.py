class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        count = 0
        curr_len = 1  # Length of current alternating subarray
        
        # Always count the first element as a subarray of length 1
        count += curr_len
        
        for i in range(1, len(nums)):
            # If the current element is different from the previous one,
            # extend the current alternating subarray by 1
            if nums[i] != nums[i - 1]:
                curr_len += 1
            else:
                # Reset the length to 1 if they are the same
                curr_len = 1
            
            count += curr_len
        
        return count