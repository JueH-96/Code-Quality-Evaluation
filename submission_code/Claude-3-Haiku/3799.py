class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        # Create a set of the unique digits
        unique_digits = set(digits)
        
        # Initialize a counter for the number of distinct 3-digit even numbers
        count = 0
        
        # Iterate through all possible 3-digit combinations
        for i in range(1, 10):
            if i in unique_digits:
                for j in unique_digits:
                    if j != i:
                        for k in unique_digits:
                            if k != i and k != j and (i * 100 + j * 10 + k) % 2 == 0:
                                count += 1
        
        return count