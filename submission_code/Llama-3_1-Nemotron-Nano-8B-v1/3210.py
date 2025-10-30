class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if s[i] in vowels else 0)
        count = 0
        for l in range(2, n + 1, 2):
            for i in range(n - l + 1):
                j = i + l
                v = prefix[j] - prefix[i]
                if v == l // 2:
                    if (v * v) % k == 0:
                        count += 1
        return count