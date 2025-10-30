class Solution:
    def smallestNumber(self, n: int) -> int:
        # The problem asks for the smallest number x such that x >= n and the binary
        # representation of x contains only set bits (i.e., is composed entirely of '1's).
        # Numbers with binary representations of all '1's are of the form 2^k - 1,
        # where k is a positive integer representing the number of '1's.
        # Examples of such numbers:
        # k=1: 2^1 - 1 = 1 (binary "1")
        # k=2: 2^2 - 1 = 3 (binary "11")
        # k=3: 2^3 - 1 = 7 (binary "111")
        # k=4: 2^4 - 1 = 15 (binary "1111")
        # ...and so on.

        # We are looking for the smallest number x >= n that belongs to this sequence (1, 3, 7, 15, ...).

        # Let's consider the number of bits in the binary representation of n.
        # Python's `bit_length()` method returns the number of bits required to represent a positive integer n
        # without leading zeros. For example:
        # n = 1 (binary "1")          -> 1.bit_length() = 1
        # n = 3 (binary "11")         -> 3.bit_length() = 2
        # n = 5 (binary "101")        -> 5.bit_length() = 3
        # n = 7 (binary "111")        -> 7.bit_length() = 3
        # n = 10 (binary "1010")      -> 10.bit_length() = 4
        k = n.bit_length()

        # Consider the number that has k bits, and all of them are set to '1'.
        # This number is 2^k - 1.
        # In binary, 2^k is represented as a '1' followed by k zeros (e.g., 2^3 = 8, binary "1000").
        # Subtracting 1 from 2^k results in a number consisting of k ones (e.g., 8 - 1 = 7, binary "111").
        # We can calculate 2^k - 1 efficiently using the bitwise left shift operator: (1 << k) - 1.
        # (1 << k) shifts the bit pattern of 1 (which is "1") left by k positions, effectively calculating 2^k.

        # Let's check if this number (1 << k) - 1 is the smallest all-ones number that is greater than or equal to n.

        # Property 1: (1 << k) - 1 is greater than or equal to n.
        # If a positive integer n has k bits, it means that its value is in the range [2^(k-1), 2^k - 1]
        # (or just [1, 1] if k=1). Specifically, n <= 2^k - 1.
        # Since (1 << k) - 1 is equal to 2^k - 1, it is guaranteed that (1 << k) - 1 >= n.

        # Property 2: Any all-ones number smaller than (1 << k) - 1 is strictly less than n.
        # An all-ones number smaller than (1 << k) - 1 must have fewer than k bits in its binary representation.
        # Let such a number be 2^m - 1, where m is the number of bits and m < k.
        # The largest possible value for 2^m - 1 with m < k occurs when m = k-1. This value is 2^(k-1) - 1.
        # Since n has k bits, n must be greater than or equal to the smallest k-bit number, which is 2^(k-1).
        # Comparing n with 2^(k-1) - 1:
        # n >= 2^(k-1)
        # 2^(k-1) > 2^(k-1) - 1
        # Therefore, n > 2^(k-1) - 1.
        # This implies that n is strictly greater than any all-ones number with fewer than k bits.

        # Conclusion:
        # The number (1 << k) - 1 (the k-bit all-ones number) is >= n (Property 1).
        # Any all-ones number smaller than (1 << k) - 1 is < n (Property 2).
        # Thus, (1 << k) - 1 is the smallest all-ones number that is greater than or equal to n.

        return (1 << k) - 1