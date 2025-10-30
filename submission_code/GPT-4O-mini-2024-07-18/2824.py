class Solution:
    def isFascinating(self, n: int) -> bool:
        # Calculate 2 * n and 3 * n
        n2 = 2 * n
        n3 = 3 * n
        
        # Concatenate n, 2*n, and 3*n as strings
        concatenated = str(n) + str(n2) + str(n3)
        
        # Check if the concatenated string contains all digits from 1 to 9 exactly once
        return sorted(concatenated) == ['1', '2', '3', '4', '5', '6', '7', '8', '9']