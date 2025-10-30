class Solution:
    def isFascinating(self, n: int) -> bool:
        two_n = 2 * n
        three_n = 3 * n
        
        # Check if both 2n and 3n are 3-digit numbers
        if len(str(two_n)) + len(str(three_n)) != 6:
            return False
        
        concatenated = str(n) + str(two_n) + str(three_n)
        
        # Check for presence of zero and all digits from 1-9 exactly once
        if '0' in concatenated:
            return False
        
        required_digits = set('123456789')
        if set(concatenated) == required_digits:
            return True
        else:
            return False