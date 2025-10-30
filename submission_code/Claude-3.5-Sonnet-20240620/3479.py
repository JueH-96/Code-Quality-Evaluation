class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        zeros = 0
        ones = 0
        left = 0
        
        for right in range(n):
            if s[right] == '0':
                zeros += 1
            else:
                ones += 1
            
            while zeros * zeros > ones:
                if s[left] == '0':
                    zeros -= 1
                else:
                    ones -= 1
                left += 1
            
            count += right - left + 1
        
        return count