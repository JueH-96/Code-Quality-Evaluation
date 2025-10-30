class Solution:
    def minimizeStringValue(self, s: str) -> str:
        count = [0] * 26
        result = list(s)
        
        for i in range(len(s)):
            if s[i] == '?':
                # Find the character with minimum count
                min_count = min(count)
                for j in range(26):
                    if count[j] == min_count:
                        result[i] = chr(ord('a') + j)
                        count[j] += 1
                        break
            else:
                count[ord(s[i]) - ord('a')] += 1
        
        return ''.join(result)