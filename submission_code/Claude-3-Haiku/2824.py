class Solution:
    def isFascinating(self, n: int) -> bool:
        # Convert n, 2*n, and 3*n to strings and concatenate them
        fascinate = str(n) + str(2*n) + str(3*n)
        
        # Check if the concatenated number contains all digits from 1 to 9 exactly once
        for i in range(1, 10):
            if str(i) not in fascinate:
                return False
        
        # Check if the concatenated number contains any 0's
        if '0' in fascinate:
            return False
        
        return True