class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # We want a subsequence where consecutive sums have the same parity.
        # Such a subsequence must either:
        # 1) Consist entirely of numbers having the same parity (all evens or all odds).
        #    In that case, each pair-sum is even.
        # 2) Alternate parity (odd-even-odd-even... or even-odd-even-odd...).
        #    In that case, each pair-sum is odd.
        #
        # Therefore, the answer is the maximum of:
        #   (a) The number of even elements, or the number of odd elements (whichever is larger),
        #   (b) The length of the longest alternating-parity subsequence.
        
        n = len(nums)
        
        # (a) Count how many are even, how many are odd
        even_count = sum(1 for x in nums if x % 2 == 0)
        odd_count = n - even_count
        same_parity_max = max(even_count, odd_count)
        
        # (b) Compute the longest alternating-parity subsequence length
        alt_count = 1  # We can always start with the first element
        last_parity = nums[0] % 2
        for i in range(1, n):
            current_parity = nums[i] % 2
            if current_parity != last_parity:
                alt_count += 1
                last_parity = current_parity
        
        # The result is the max of the two possibilities
        return max(same_parity_max, alt_count)