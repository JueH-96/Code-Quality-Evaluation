class Solution:
    def minimizeStringValue(self, s: str) -> str:
        freq = [0] * 26
        modify = []
        ans = []
        
        for ch in s:
            if ch != '?':
                freq[ord(ch) - ord('a')] += 1
            else:
                modify.append(len(ans))
            ans.append('?')
        
        modify.sort() # Sort indices where modification is needed
        
        for idx in modify:
            best = 0
            for i in range(26):
                if freq[i] < freq[best]:
                    best = i
                elif freq[i] == freq[best] and i < best:
                    best = i
            ans[idx] = chr(best + ord('a'))
            freq[best] += 1
        
        return ''.join(ans)