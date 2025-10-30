class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        # Precompute prefix sums of vowels
        prefix_vowels = [0] * (n + 1)
        for i in range(n):
            prefix_vowels[i + 1] = prefix_vowels[i] + (1 if s[i] in vowels else 0)
        
        # Collect all valid m values where m^2 mod k == 0
        max_m = n // 2
        valid_m = []
        for m in range(1, max_m + 1):
            if (m * m) % k == 0:
                valid_m.append(m)
        
        res = 0
        # For each valid m, check all possible substrings of length 2m
        for m in valid_m:
            l = 2 * m
            if l > n:
                continue
            for i in range(n - l + 1):
                end = i + l
                vowels_count = prefix_vowels[end] - prefix_vowels[i]
                if vowels_count == m:
                    res += 1
        
        return res