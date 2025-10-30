class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count the number of even and odd elements
        even_count = sum(1 for x in nums if x % 2 == 0)
        odd_count = len(nums) - even_count
        
        # Option 1: All elements the same parity (sum is always even)
        # The subsequence is either all evens or all odds
        same_parity_longest = max(even_count, odd_count)
        
        # Option 2: An alternating parity subsequence (sum is always odd)
        # We can greedily build the longest alternating subsequence
        # by adding elements that alternate in parity from the last chosen
        longest_alternating = 0
        prev_parity = None
        for x in nums:
            p = x % 2
            if prev_parity is None or p != prev_parity:
                longest_alternating += 1
                prev_parity = p
        
        # The answer is the maximum between these two options
        return max(same_parity_longest, longest_alternating)