class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        vowels = set('aeiou')
        # prefix_v[i] = number of vowels in s[:i]
        prefix_v = [0] * (n + 1)
        for i, ch in enumerate(s):
            prefix_v[i + 1] = prefix_v[i] + (1 if ch in vowels else 0)

        count = 0
        # Enumerate all substrings s[i:j]
        # Only substrings of even length can have vowels == consonants.
        for i in range(n):
            # start j from i+2 (minimum even length 2), step by 2
            for L in range(2, n - i + 1, 2):
                j = i + L
                v = prefix_v[j] - prefix_v[i]
                c = L - v
                # v == c (by construction L is even, but still check)
                # and (v * c) % k == 0
                if v == c and (v * c) % k == 0:
                    count += 1

        return count