class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        n = len(s)
        s_indices = {}
        t_indices = {}
        for i in range(n):
            s_indices[s[i]] = i
            t_indices[t[i]] = i
        
        diff = 0
        for char in s:
            diff += abs(s_indices[char] - t_indices[char])
        
        return diff