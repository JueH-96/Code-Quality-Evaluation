class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        prefix_v = [0] * (n + 1)
        vowels = {'a', 'e', 'i', 'o', 'u'}
        for i in range(n):
            prefix_v[i+1] = prefix_v[i] + (1 if s[i] in vowels else 0)
        count = 0
        for i in range(n):
            for j in range(i, n):
                v = prefix_v[j+1] - prefix_v[i]
                c = (j - i + 1) - v
                if v == c and (v * c) % k == 0:
                    count += 1
        return count