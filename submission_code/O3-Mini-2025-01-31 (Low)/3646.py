from typing import List
from collections import defaultdict

MOD = 10**9 + 7

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        # We'll use two dictionaries:
        #   cnt[val] = number of good subsequences (in order) ending with value 'val'
        #   sm[val] = total sum of elements across all those good subsequences ending with 'val'
        #
        # When processing a new number a, it can either start its own subsequence,
        #   or extend an existing subsequence if the previous ending element is either a-1 or a+1.
        # Extending a subsequence ending in neighbor 'nb' creates a new subsequence with:
        #   sum = (oldSum + (oldCount * a))  because we append a to each subsequence.
        # And the count is increased by oldCount.
        #
        # Each element always introduces a new subsequence (of single element a) with sum = a.
        cnt = defaultdict(int)
        sm = defaultdict(int)
        
        for a in nums:
            # Start a new subsequence with this number.
            new_count = 1
            new_sum = a  # subsequence [a]
            # Check neighbor a-1: if present, extend each subsequence ending with a-1.
            if (a - 1) in cnt:
                new_count = (new_count + cnt[a - 1]) % MOD
                new_sum = (new_sum + sm[a - 1] + cnt[a - 1] * a) % MOD
            # Check neighbor a+1: if present, extend each subsequence ending with a+1.
            if (a + 1) in cnt:
                new_count = (new_count + cnt[a + 1]) % MOD
                new_sum = (new_sum + sm[a + 1] + cnt[a + 1] * a) % MOD
            # Add the new subsequences we formed ending in `a`.
            cnt[a] = (cnt[a] + new_count) % MOD
            sm[a] = (sm[a] + new_sum) % MOD
        
        # The result is the sum over all subsequences (i.e. over all ending values).
        result = sum(sm.values()) % MOD
        return result

# For quick local testing.
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.sumOfGoodSubsequences([1, 2, 1]))  # Expected output: 14
    # Example 2
    print(sol.sumOfGoodSubsequences([3, 4, 5]))  # Expected output: 40