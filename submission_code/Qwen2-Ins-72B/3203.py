from collections import Counter
from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        prefix_sum = [Counter()]
        for i, char in enumerate(s):
            prefix_sum.append(prefix_sum[-1].copy())
            prefix_sum[-1][char] += 1

        def count_mismatch(a, b):
            return sum((prefix_sum[b + 1] - prefix_sum[a]).values()) // 2

        answer = []
        for a, b, c, d in queries:
            mismatch_left = count_mismatch(a, b)
            mismatch_right = count_mismatch(c - n // 2, d - n // 2)
            answer.append(mismatch_left == mismatch_right)

        return answer