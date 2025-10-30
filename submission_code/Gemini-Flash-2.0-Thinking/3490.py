class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n

        count_even = 0
        count_odd = 0
        for num in nums:
            if num % 2 == 0:
                count_even += 1
            else:
                count_odd += 1

        # Case 1: Subsequence with the same parity
        max_length_same_parity = max(count_even, count_odd)

        # Case 2: Subsequence with alternating parity
        max_length_alternating = 0
        if count_even > 0 and count_odd > 0:
            max_length_alternating = count_even + count_odd

        return max(max_length_same_parity, max_length_alternating)