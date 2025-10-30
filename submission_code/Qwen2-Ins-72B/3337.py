class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        n = len(s)
        c_indices = [i for i, char in enumerate(s) if char == c]
        total = 0
        
        for i in c_indices:
            left = right = i
            while left >= 0 and right < n and s[left] == s[right]:
                total += 1
                left -= 1
                right += 1
                
        return total