class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        left = 0
        zeros = 0
        count_less_than_200 = 0
        for right in range(n):
            if s[right] == '0':
                zeros += 1
            while zeros > 199:
                if s[left] == '0':
                    zeros -= 1
                left += 1
            count_less_than_200 += (right - left + 1)
        
        type2_non_dominant = 0
        for left in range(n):
            z = 0
            for right in range(left, n):
                if s[right] == '0':
                    z += 1
                if z >= 200:
                    break
                length = right - left + 1
                ones = length - z
                if ones < z * z:
                    type2_non_dominant += 1
        
        return count_less_than_200 - type2_non_dominant