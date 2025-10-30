class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        if n < 2:
            return n
        # Initialize the longest valid subsequence length
        max_len = 1
        # Initialize the current valid subsequence length
        current_len = 1
        # Determine the parity of the first element
        prev_parity = nums[0] % 2
        # Iterate through the array starting from the second element
        for i in range(1, n):
            current_parity = nums[i] % 2
            # Check if the current pair (prev_parity, current_parity) is valid
            if (prev_parity + current_parity) % 2 == (nums[i-1] % 2 + nums[i] % 2) % 2:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                # Reset current_len to 1 if the current pair is not valid
                current_len = 1
            # Update prev_parity
            prev_parity = current_parity
        return max_len