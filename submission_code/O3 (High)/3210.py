class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        # Set of vowels for quick lookup
        vowels = set('aeiou')
        n = len(s)

        # prefix_v[i] = number of vowels in s[0 : i]
        prefix_v = [0]*(n+1)
        for i, ch in enumerate(s):
            prefix_v[i+1] = prefix_v[i] + (ch in vowels)

        beautiful = 0
        # Enumerate all substrings s[l : r]   (0-based, r inclusive)
        for l in range(n):
            for r in range(l, n):
                v = prefix_v[r+1] - prefix_v[l]          # vowels in substring
                c = (r - l + 1) - v                      # consonants in substring
                if v == c and (v * c) % k == 0:          # v == c ⇒ product = v²
                    beautiful += 1

        return beautiful