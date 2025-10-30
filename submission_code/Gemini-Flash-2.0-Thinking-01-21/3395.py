import math

class Solution:
    def minAnagramLength(self, s: str) -> int:
        counts = {}
        for char in s:
            counts[char] = counts.get(char, 0) + 1
        
        if not counts:
            return 0
            
        count_values = list(counts.values())
        
        if not count_values:
            return 0
            
        common_divisor = count_values[0]
        for i in range(1, len(count_values)):
            common_divisor = math.gcd(common_divisor, count_values[i])
            
        min_length = 0
        for count in count_values:
            min_length += count // common_divisor
            
        return min_length