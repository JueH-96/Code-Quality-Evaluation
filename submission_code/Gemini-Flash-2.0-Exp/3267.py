class Solution:
    def maximumLength(self, s: str) -> int:
        counts = {}
        for char in set(s):
            counts[char] = []
        
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and s[i] == s[j]:
                j += 1
            length = j - i
            counts[s[i]].append(length)
            i = j
        
        ans = -1
        for char in counts:
            counts[char].sort(reverse=True)
            
            if len(counts[char]) >= 3:
                ans = max(ans, counts[char][2])
            elif len(counts[char]) == 2:
                ans = max(ans, min(counts[char][0] - 1, counts[char][1]))
            elif len(counts[char]) == 1:
                ans = max(ans, counts[char][0] - 2)
        
        return ans