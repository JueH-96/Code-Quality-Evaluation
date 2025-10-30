from typing import List
from collections import Counter

MOD = 10**9 + 7

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        # Given sum(nums) <= 20000, we let S be the total sum.
        S = sum(nums)
        # dp[s] will count number of ways to form sub-multiset with sum s.
        dp = [0] * (S + 1)
        dp[0] = 1
        
        # Count frequency for each unique number
        freq = Counter(nums)
        
        # Process each unique number and its count.
        for x, count in freq.items():
            new_dp = [0] * (S + 1)
            
            if x == 0:
                # For x==0, adding any number of 0's (from 0 to count) doesn't change the sum.
                # So, every dp[s] becomes dp[s]*(count+1)
                for s in range(S + 1):
                    new_dp[s] = dp[s] * (count + 1) % MOD
            else:
                # We'll use a sliding window (prefix sum) method to do convolution
                prefix = [0] * (S + 1)
                prefix[0] = dp[0]
                for s in range(1, S + 1):
                    prefix[s] = (prefix[s - 1] + dp[s]) % MOD
                
                # For every possible sum s, we want:
                # new_dp[s] = dp[s] + dp[s - x] + dp[s - 2*x] + ... + dp[s - k*x]
                # where k can be up to count and s - k*x >= 0.
                # Using prefix sums, we can compute:
                # new_dp[s] = prefix[s] - (prefix[s - (count+1)*x] if s - (count+1)*x >= 0 else 0)
                for s in range(S + 1):
                    # The lower bound index for valid dp values is s - (count*x)
                    # But we need to subtract out extra terms if k > count.
                    low_idx = s - (count + 1) * x
                    total = prefix[s]
                    if low_idx >= 0:
                        total = (total - prefix[low_idx] + MOD) % MOD
                    new_dp[s] = total
            dp = new_dp

        # Sum up dp[s] for s in [l, r] (making sure s does not exceed S)
        result = 0
        # Clamp r so that we don't index out of bounds.
        r_bound = min(r, S)
        for s in range(l, r_bound + 1):
            result = (result + dp[s]) % MOD
        
        return result