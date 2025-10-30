class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if s[i] in vowels else 0)
        count = 0
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                if length % 2 == 0:
                    required = length // 2
                    vowels_count = prefix[j+1] - prefix[i]
                    if vowels_count == required:
                        if (vowels_count * vowels_count) % k == 0:
                            count += 1
        return count