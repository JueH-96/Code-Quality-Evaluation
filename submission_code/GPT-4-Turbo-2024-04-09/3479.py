class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        count = 0
        
        # Iterate over all possible starting points of substrings
        for i in range(n):
            num_zeros = 0
            num_ones = 0
            
            # Iterate over all possible ending points of substrings starting from i
            for j in range(i, n):
                if s[j] == '0':
                    num_zeros += 1
                else:
                    num_ones += 1
                
                # Check if the number of ones is greater than or equal to the square of the number of zeros
                if num_ones >= num_zeros ** 2:
                    count += 1
        
        return count