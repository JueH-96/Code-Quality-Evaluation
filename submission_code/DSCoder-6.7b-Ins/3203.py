from collections import Counter

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] ^ (1 << (ord(s[i]) - ord('a')))

        def check(l, r, k):
            hash_val = prefix[r + 1] ^ prefix[l]
            bit_count = bin(hash_val).count('1')
            return bit_count // 2 <= k

        return [check(a, b, (b - a) // 2) for a, b, _, _ in queries]