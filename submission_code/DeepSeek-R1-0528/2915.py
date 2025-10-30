from collections import defaultdict

class Solution:
    def countInterestingSubarrays(self, nums: list, modulo: int, k: int) -> int:
        cur = 0
        freq = defaultdict(int)
        freq[0] = 1
        res = 0
        for num in nums:
            cur += (num % modulo == k)
            R = cur % modulo
            target = (R - k) % modulo
            res += freq[target]
            freq[R] += 1
        return res