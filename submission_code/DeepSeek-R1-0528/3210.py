from collections import defaultdict

class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        max_x = n // 2
        valid = [False] * (max_x + 1)
        for x in range(1, max_x + 1):
            if (x * x) % k == 0:
                valid[x] = True
        
        d = defaultdict(lambda: [[], []])
        d[0][0].append(0)
        prefix = 0
        res = 0
        vowels = "aeiou"
        
        for j in range(1, n + 1):
            char = s[j - 1]
            if char in vowels:
                prefix += 1
            else:
                prefix -= 1
            current_parity = j % 2
            lst = d[prefix][current_parity]
            for i in lst:
                length = j - i
                x_val = length // 2
                if x_val <= max_x and valid[x_val]:
                    res += 1
            d[prefix][current_parity].append(j)
        
        return res