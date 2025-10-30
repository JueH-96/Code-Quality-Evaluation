class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        from collections import Counter
        from itertools import accumulate
        
        MOD = 10**9 + 7
        
        # Count occurrences of each number
        count = Counter(nums)
        unique_nums = list(count.keys())
        
        # Prepare a dp array where dp[i] means the number of ways to form sum i
        max_sum = sum(num * count[num] for num in unique_nums)
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # There's one way to make sum 0: the empty set
        
        # Fill the dp array
        for num in unique_nums:
            occurrences = count[num]
            for j in range(max_sum, num - 1, -1):
                for k in range(1, occurrences + 1):
                    if j >= k * num:
                        dp[j] = (dp[j] + dp[j - k * num]) % MOD
                    else:
                        break
        
        # Calculate the total number of valid sub-multisets
        result = sum(dp[l:r + 1]) % MOD
        
        return result