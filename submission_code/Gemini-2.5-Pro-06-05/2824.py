class Solution:
    def isFascinating(self, n: int) -> bool:
        """
        Checks if a 3-digit number n is "fascinating".

        A number n is fascinating if the concatenation of n, 2*n, and 3*n
        contains all digits from 1 to 9 exactly once.
        """
        
        # Step 1: Form the concatenated string from n, 2*n, and 3*n.
        # Using an f-string is a clean and modern way to do this.
        # e.g., for n = 192, s becomes "192384576"
        s = f"{n}{2*n}{3*n}"
        
        # Step 2: Check if the string meets the "fascinating" criteria.
        # A fascinating string must be a permutation of "123456789".
        # This implies two conditions:
        # 1. The string must have a length of 9.
        # 2. The set of characters in the string must be {'1', ..., '9'}.
        #
        # The second condition ensures there are no '0's and all required digits
        # are present. When combined with the first condition (length is 9),
        # it also implicitly guarantees that there are no duplicate digits.
        
        return len(s) == 9 and set(s) == {'1', '2', '3', '4', '5', '6', '7', '8', '9'}