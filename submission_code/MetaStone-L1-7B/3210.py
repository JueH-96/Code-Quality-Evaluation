class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        if n == 0:
            return 0
        prefix_vowels = [0] * (n + 1)
        vowels = {'a', 'e', 'i', 'o', 'u'}
        for i in range(n):
            prefix_vowels[i+1] = prefix_vowels[i] + (1 if s[i] in vowels else 0)
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                if length % 2 != 0:
                    continue
                vowels_count = prefix_vowels[j+1] - prefix_vowels[i]
                if vowels_count != (length // 2):
                    continue
                if (vowels_count ** 2) % k == 0:
                    count += 1
        return count