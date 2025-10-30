from typing import List
from math import gcd

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        count = 0

        for i in range(1, 1 << n):
            for j in range(1, 1 << n):
                if (i & j) == 0:  # Check if subsequences are disjoint
                    indices1 = [k for k in range(n) if (i >> k) & 1]
                    indices2 = [k for k in range(n) if (j >> k) & 1]

                    if not indices1 or not indices2:
                        continue

                    subseq1 = [nums[k] for k in indices1]
                    subseq2 = [nums[k] for k in indices2]

                    def calculate_gcd(subsequence):
                        if not subsequence:
                            return 0
                        result = subsequence[0]
                        for k in range(1, len(subsequence)):
                            result = gcd(result, subsequence[k])
                        return result

                    gcd1 = calculate_gcd(subseq1)
                    gcd2 = calculate_gcd(subseq2)

                    if gcd1 == gcd2:
                        count = (count + 1) % MOD

        return count