class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        from collections import Counter
        MOD = 10**9 + 7
        
        # Count frequency of each number
        freq = Counter(nums)
        
        # Maximum possible sum of elements
        max_sum = sum(nums)
        
        # dp[s] will be the number of sub-multisets with sum exactly s
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # There's one way to make sum 0: use the empty set
        
        # Iterate over each unique number in the array
        for num, count in freq.items():
            # Update dp array from the back to avoid overwriting results of the same step
            for s in range(max_sum, num - 1, -1):
                # We can add this number between 1 to count times
                # dp[s] should be incremented by dp[s-num], dp[s-2*num], ..., dp[s-count*num]
                # if those indices are valid
                for k in range(1, count + 1):
                    if s - k * num >= 0:
                        dp[s] = (dp[s] + dp[s - k * num]) % MOD
                    else:
                        break
        
        # Calculate the sum of all valid dp[s] where l <= s <= r
        result = sum(dp[s] for s in range(l, r + 1)) % MOD
        
        return result