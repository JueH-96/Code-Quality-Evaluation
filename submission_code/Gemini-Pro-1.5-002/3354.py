class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        ans = ""
        counts = [0] * 26
        q_indices = []

        for i in range(n):
            if s[i] == '?':
                q_indices.append(i)
            else:
                ans += s[i]
                counts[ord(s[i]) - ord('a')] += 1
        
        q_count = len(q_indices)
        
        
        char_index = 0
        res = list(ans)
        for i in q_indices:
            while char_index < 26 and counts[char_index] > 0:
                char_index += 1
            
            if char_index < 26:
                res[i] = chr(ord('a') + char_index)
                counts[char_index] += 1
        
        return "".join(res)