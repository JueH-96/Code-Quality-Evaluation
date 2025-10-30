class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        if n < 2:
            return 0
        
        pre = [0] * (n + 1)
        vowels_set = set('aeiou')
        
        for i in range(1, n + 1):
            if s[i - 1] in vowels_set:
                pre[i] = pre[i - 1] + 1
            else:
                pre[i] = pre[i - 1]
        
        count = 0
        for i in range(n):
            for j in range(i + 1, n, 2):
                L = j - i + 1
                vowels_count = pre[j + 1] - pre[i]
                if 2 * vowels_count == L:
                    if (vowels_count * vowels_count) % k == 0:
                        count += 1
        return count