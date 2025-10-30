class Solution:
    def minimumLength(self, s: str) -> int:
        left = 0
        right = len(s) - 1
        
        while left < right and s[left] == s[right]:
            c = s[left]
            l_count = 0
            r_count = 0
            l_temp = left
            r_temp = right
            
            while l_temp <= right and s[l_temp] == c:
                l_count += 1
                l_temp += 1
            while r_temp >= left and s[r_temp] == c:
                r_count += 1
                r_temp -= 1
            
            if l_temp > r_temp:
                return 0
            if l_count + r_count < 3 and l_temp <= r_temp:
                break
            
            left = l_temp
            right = r_temp
        
        return right - left + 1