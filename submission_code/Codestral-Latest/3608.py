from math import gcd
from itertools import combinations
from functools import reduce
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7

        def all_subsequences(arr):
            n = len(arr)
            subsequences = []
            for i in range(1, 1 << n):
                subsequence = [arr[j] for j in range(n) if (i & (1 << j))]
                subsequences.append(subsequence)
            return subsequences

        def gcd_of_list(lst):
            return reduce(gcd, lst)

        n = len(nums)
        all_subs = all_subsequences(nums)
        count = 0

        for i in range(len(all_subs)):
            for j in range(i + 1, len(all_subs)):
                seq1 = all_subs[i]
                seq2 = all_subs[j]
                if not set(seq1) & set(seq2):
                    if gcd_of_list(seq1) == gcd_of_list(seq2):
                        count += 1

        return count % MOD