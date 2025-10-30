class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        if n < 2:
            return n
        # Initialize the longest subsequence length
        max_len = 1
        # Initialize the current subsequence length
        current_len = 1
        # Initialize the parity of the last pair
        last_parity = (nums[0] + nums[1]) % 2
        current_len = 2
        for i in range(2, n):
            current_parity = (nums[i-1] + nums[i]) % 2
            if current_parity == last_parity:
                current_len += 1
            else:
                if current_len > max_len:
                    max_len = current_len
                current_len = 2
                last_parity = current_parity
        if current_len > max_len:
            max_len = current_len
        return max_len