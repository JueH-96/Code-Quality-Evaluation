class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        count_zeros = s.count('0')
        count_ones = len(s) - count_zeros
        dominant_count = 0
        
        for i in range(len(s)):
            if s[i] == '0':
                dominant_count += count_ones * (count_ones + 1) // 2
                count_ones -= 1
            else:
                dominant_count += 1
                count_zeros -= 1
        
        return dominant_count