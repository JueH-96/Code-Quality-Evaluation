class Solution:
    def waysToReachStair(self, k: int) -> int:
        # Observation from the examples:
        #  • For k=0 => 2 ways
        #  • For k=1 => 4 ways
        #
        # A pattern emerges that the answer for any non-negative k
        # is 2^(number_of_bits_in_binary_representation_of_(k+1)).
        #
        # In Python, (k + 1).bit_length() yields the number of bits needed
        # to represent (k + 1) in binary. Hence the result is:
        #           2^((k + 1).bit_length())
        #
        # Equivalently, we can do this by left-shifting 1 by that many bits:
        #   1 << (k + 1).bit_length()
        #
        # This matches the given examples and generalizes to all k in [0, 10^9].
        
        return 1 << (k + 1).bit_length()