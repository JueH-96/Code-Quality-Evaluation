from typing import List
from collections import Counter

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        mod = 10**9 + 7
        # Count frequencies of each number.
        freq = Counter(nums)
        # Separate out the zeros because they don't change the sum.
        zero_count = freq.get(0, 0)
        if 0 in freq:
            del freq[0]
        # For nonzero numbers, group them as (value, count). (Order doesn’t matter.)
        groups = sorted(freq.items(), key=lambda x: x[0])
        
        # dp[s] will represent the number of ways (from nonzero groups processed so far)
        # to achieve a sum of s. Initially only the empty multiset is possible.
        dp = [1]  # dp[0] = 1
        old_max = 0  # maximum sum that we can create so far (from nonzero groups)
        
        # Process each group. For a given number v with frequency c,
        # the generating function is: 1 + x^(v) + x^(2*v) + ... + x^(c*v)
        # We wish to update:
        #   new_dp[s] = dp[s] + dp[s - v] + dp[s - 2*v] + ... (for s - j*v >= 0, j <= c)
        # A standard recurrence (using sliding window / prefix sums over steps of v)
        # is:
        #   new_dp[s] = new_dp[s-v] + (dp[s] if s exists) - (dp[s - (c+1)*v] if valid)
        # (Taking care that dp[s] is defined if s <= old_max and 0 otherwise.)
        for v, c in groups:
            new_max = old_max + v * c
            new_dp = [0] * (new_max + 1)
            # Instead of doing a nested loop, we use a recurrence which is equivalent to:
            # new_dp[s] = (new_dp[s - v] if s >= v else 0)
            #             + (dp[s] if s <= old_max else 0)
            #             - (dp[s - v*(c+1)] if s - v*(c+1) is valid, i.e. >= 0 and within old_max)
            limit = v * (c + 1)  # when s >= limit, we subtract dp[s - limit]
            for s in range(new_max + 1):
                if s < v:
                    # For s < v, we cannot use any copy of v because even one copy would add v.
                    new_dp[s] = dp[s] if s <= old_max else 0
                else:
                    temp = new_dp[s - v]
                    if s <= old_max:
                        temp += dp[s]
                    # Subtract the extra term if it exists.
                    if s >= limit and (s - limit) <= old_max:
                        temp -= dp[s - limit]
                    new_dp[s] = temp % mod
            dp = new_dp
            old_max = new_max
        
        # Now, dp[s] represents the count of ways (from nonzero numbers) to achieve sum s.
        # Sums greater than old_max are not possible.
        total = 0
        low_bound = l
        high_bound = min(r, old_max)
        for s in range(low_bound, high_bound + 1):
            total = (total + dp[s]) % mod
        
        # Zeros can be chosen in any count from 0 to zero_count without affecting the sum.
        # Thus, for each nonzero selection, we have (zero_count+1) distinct choices.
        total = (total * (zero_count + 1)) % mod
        return total