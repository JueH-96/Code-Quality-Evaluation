class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        ones = 0
        zeros = 0
        left = 0
        
        for right in range(n):
            if s[right] == '1':
                ones += 1
            else:
                zeros += 1
            
            while ones < zeros ** 2:
                if s[left] == '1':
                    ones -= 1
                else:
                    zeros -= 1
                left += 1
            
            count += right - left + 1
        
        return count