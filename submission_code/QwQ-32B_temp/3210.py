class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        # Compute prefix sum of vowels
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + (1 if s[i] in vowels else 0)
        
        # Precompute valid lengths
        valid = [False] * (n + 2)  # valid[L] for L up to n
        for L in range(2, n + 1, 2):
            m = L // 2
            if (m * m) % k == 0:
                valid[L] = True
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                L = j - i + 1
                if L % 2 != 0:
                    continue
                if not valid[L]:
                    continue
                v = prefix[j + 1] - prefix[i]
                if v == L // 2:
                    count += 1
        return count