class Solution:
    def isFascinating(self, n: int) -> bool:
        """
        Checks if an integer n (consisting of exactly 3 digits) is fascinating.
        A number n is fascinating if, after concatenating n, 2*n, and 3*n,
        the resulting number contains all digits from 1 to 9 exactly once
        and does not contain any 0's.
        """
        # Calculate 2*n and 3*n
        double_n = 2 * n
        triple_n = 3 * n

        # Convert n, 2*n, and 3*n to strings and concatenate them
        s_concat = str(n) + str(double_n) + str(triple_n)

        # To check if the concatenated string contains all digits 1-9 exactly once
        # and does not contain any 0's, we can use a set to track seen digits.
        
        seen_digits = set()
        for char in s_concat:
            # Condition: does not contain any 0's
            if char == '0':
                return False
            
            # Condition: contains all the digits from 1 to 9 exactly once
            # Check for duplicate non-zero digits. If we see a digit we've already seen,
            # it means the digit is repeated, violating the "exactly once" rule.
            if char in seen_digits:
                return False
                
            # Add the unique non-zero digit to the set
            seen_digits.add(char)

        # After iterating through the entire concatenated string:
        # 1. We've confirmed it doesn't contain '0'.
        # 2. We've confirmed all non-zero digits encountered were unique.
        # Now, we need to verify that all digits from '1' to '9' were present.
        # The size of the 'seen_digits' set should be exactly 9 if all digits
        # from '1' through '9' were present and unique.
        # If the concatenated string had a length other than 9, it's impossible
        # to collect exactly 9 unique digits from 1-9, so the set size check
        # also implicitly handles the overall length requirement for a fascinating number.
        return len(seen_digits) == 9