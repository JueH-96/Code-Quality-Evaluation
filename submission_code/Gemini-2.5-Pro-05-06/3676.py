class Solution:
    def smallestNumber(self, n: int) -> int:
        """
        Finds the smallest number x >= n such that the binary representation of x
        contains only set bits (1s).

        Numbers with all set bits in binary are of the form 2^k - 1:
        1 (binary "1")
        3 (binary "11")
        7 (binary "111")
        15 (binary "1111")
        ...

        We can generate these numbers iteratively and find the first one >= n.
        Start with x = 1.
        In each step, if x < n, update x to the next number with all set bits.
        The next number with all set bits after x (where x = 2^k - 1) is 2^(k+1) - 1.
        This can be calculated as (x << 1) | 1.
        Example: if x = 3 (binary "11"), then x << 1 = 6 (binary "110").
                 (x << 1) | 1 = 6 | 1 = 7 (binary "111").
        """
        
        # Initialize x to the smallest number with all set bits (which is 1).
        # Note: n is a positive number, so n >= 1.
        current_num_all_ones = 1
        
        # Keep finding the next number with all set bits until 
        # current_num_all_ones is greater than or equal to n.
        while current_num_all_ones < n:
            # Calculate the next number in the sequence 1, 3, 7, 15, ...
            # If current_num_all_ones is '11...1' (k ones),
            # current_num_all_ones << 1 is '11...10' (k ones followed by a zero),
            # (current_num_all_ones << 1) | 1 is '11...11' (k+1 ones).
            current_num_all_ones = (current_num_all_ones << 1) | 1
            
        return current_num_all_ones