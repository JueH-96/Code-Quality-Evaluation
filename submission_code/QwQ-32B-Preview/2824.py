class Solution:
    def isFascinating(self, n: int) -> bool:
        # Compute 2*n and 3*n
        double_n = 2 * n
        triple_n = 3 * n
        
        # Concatenate n, 2*n, and 3*n as strings
        concatenated = str(n) + str(double_n) + str(triple_n)
        
        # Check if length is exactly 9
        if len(concatenated) != 9:
            return False
        
        # Check if there are any zeros
        if '0' in concatenated:
            return False
        
        # Check if all digits from 1 to 9 are present exactly once
        expected_digits = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        return set(concatenated) == expected_digits