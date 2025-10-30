class Solution:
    def minimizeStringValue(self, s: str) -> str:
        res = list(s)
        freq = [0] * 26
        
        for i in range(len(res)):
            if res[i] == '?':
                min_freq = float('inf')
                best_char = None
                for j in range(26):
                    count = freq[j]
                    c = chr(ord('a') + j)
                    if count < min_freq:
                        min_freq = count
                        best_char = c
                    elif count == min_freq and c < best_char:
                        best_char = c
                res[i] = best_char
                idx_best = ord(best_char) - ord('a')
                freq[idx_best] += 1
            else:
                idx_known = ord(res[i]) - ord('a')
                freq[idx_known] += 1
                
        return ''.join(res)