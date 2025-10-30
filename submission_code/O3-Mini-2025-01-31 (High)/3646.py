from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        # dp will map a value v to a tuple (count, totalSum)
        # where count is the total number of good subsequences (formed so far)
        # that end with the value v, and totalSum is the sum of the sums
        # of all these subsequences.
        dp = {}
        # Global accumulator for the sum of all good subsequences
        ans = 0
        
        for x in nums:
            # We always can start a good subsequence with a single element x.
            # It has count = 1 and its sum is just x.
            new_count = 1
            new_sum = x % MOD
            
            # For a subsequence to be good, if we extend an existing subsequence
            # whose last element is y, then we require |y - x| == 1.
            # Thus, check the two possible neighbor values: (x-1) and (x+1).
            for neighbor in (x-1, x+1):
                cnt, sm = dp.get(neighbor, (0, 0))
                # Each good subsequence ending in 'neighbor' can be extended with x.
                # Their new sum becomes (old subsequence sum) + (x added to each sequence).
                new_count = (new_count + cnt) % MOD
                new_sum = (new_sum + sm + cnt * x) % MOD
            
            # Update dp[x]: append these newly formed subsequences to any
            # that have ended with x before.
            old_count, old_sum = dp.get(x, (0, 0))
            dp[x] = ((old_count + new_count) % MOD, (old_sum + new_sum) % MOD)
            
            # Add the sum (of all newly created subsequences ending at x)
            # to our global answer.
            ans = (ans + new_sum) % MOD
            
        return ans

# For example testing:
if __name__ == '__main__':
    sol = Solution()
    print(sol.sumOfGoodSubsequences([1,2,1]))  # Expected output: 14
    print(sol.sumOfGoodSubsequences([3,4,5]))  # Expected output: 40