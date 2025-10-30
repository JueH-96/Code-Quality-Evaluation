class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n and 3*n
        concat = str(n) + str(2*n) + str(3*n)
        
        # Check if the length of the concatenated string is 9 (since we need digits from 1 to 9)
        if len(concat) != 9:
            return False
        
        # Check if the concatenated string contains all digits from 1 to 9 exactly once
        for digit in '123456789':
            if digit not in concat:
                return False
                
        return True