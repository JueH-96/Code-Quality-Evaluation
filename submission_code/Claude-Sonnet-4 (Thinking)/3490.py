class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Case 1: All same parity (all consecutive sums are even)
        even_count = sum(1 for x in nums if x % 2 == 0)
        odd_count = len(nums) - even_count
        same_parity = max(even_count, odd_count)
        
        # Case 2: Alternating parity (all consecutive sums are odd)
        # dp[0] = length of longest alternating subsequence ending with even
        # dp[1] = length of longest alternating subsequence ending with odd
        dp = [0, 0]
        
        for num in nums:
            parity = num % 2
            if parity == 0:  # even
                dp[0] = max(dp[0], dp[1] + 1)
            else:  # odd
                dp[1] = max(dp[1], dp[0] + 1)
        
        alternating = max(dp)
        
        return max(same_parity, alternating)