class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if s[i] in vowels else 0)
        count = 0
        for l in range(2, n + 1, 2):
            m = l // 2
            if (m * m) % k != 0:
                continue
            max_i = n - l
            for i in range(max_i + 1):
                j = i + l
                if prefix[j] - prefix[i] == m:
                    count += 1
        return count