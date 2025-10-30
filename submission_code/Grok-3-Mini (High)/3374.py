class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0  # Though constraints ensure n >= 1, this handles edge case
        ans = 1  # Start with the first element's subarray
        curr_len = 1  # Length of the longest alternating subarray ending at current index
        for i in range(1, n):
            if nums[i] == nums[i - 1]:  # If current element is same as previous
                curr_len = 1  # Reset length to 1
            else:  # If current element is different from previous
                curr_len += 1  # Increase length by 1
            ans += curr_len  # Add the number of alternating subarrays ending at index i
        return ans