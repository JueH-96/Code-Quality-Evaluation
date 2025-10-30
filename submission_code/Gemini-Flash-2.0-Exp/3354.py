class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        res = list(s)
        for i in range(n):
            if res[i] == '?':
                counts = {}
                for char in 'abcdefghijklmnopqrstuvwxyz':
                    counts[char] = 0
                
                for j in range(i):
                    if res[j] != '?':
                        counts[res[j]] += 1
                
                min_count = float('inf')
                best_char = ''
                for char in 'abcdefghijklmnopqrstuvwxyz':
                    if counts[char] < min_count:
                        min_count = counts[char]
                        best_char = char
                
                res[i] = best_char
        
        return "".join(res)