class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n with 2*n and 3*n
        concatenated = str(n) + str(2*n) + str(3*n)
        
        # Check if the length is 9 (all digits from 1 to 9)
        if len(concatenated) != 9:
            return False
        
        # Check if all digits from 1 to 9 are present exactly once
        digit_set = set(concatenated)
        return len(digit_set) == 9 and '0' not in digit_set