class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Check if k can be obtained from n by only changing 1s to 0s.
        # This is possible if and only if every bit that is 1 in k is also 1 in n.
        # This condition can be checked using a bitwise AND: (n & k) must be equal to k.
        # If there is any bit position where k has a 1 and n has a 0, then (n & k) will
        # have a 0 at that position, making (n & k) different from k.
        # In this scenario, it's impossible to make n equal to k by only changing 1s to 0s,
        # because we cannot change a 0 in n to a 1.
        if (n & k) != k:
            return -1

        # If the condition above is met ((n & k) == k), it means that for every bit where
        # k has a 1, n also has a 1. We can potentially make n equal to k.
        # The changes required are to turn the 1s in n into 0s at the positions where
        # k has a 0 and n has a 1.
        # Consider the bitwise XOR of n and k: (n ^ k).
        # This operation results in a number where bits are set (1) at positions where
        # the corresponding bits in n and k are different, and 0 otherwise.
        # Since we already know that we cannot have a difference where n_i=0 and k_i=1
        # (because (n & k) == k implies n_i must be 1 if k_i is 1),
        # the only possible differences are where n_i=1 and k_i=0.
        # These positions are exactly the bits in n that need to be changed from 1 to 0
        # to match the 0s in k.
        # The number of such changes is the number of set bits (population count) in (n ^ k).

        # Calculate the XOR difference between n and k
        diff = n ^ k

        # Count the number of set bits (1s) in the result of the XOR operation.
        # This count represents the minimum number of changes (1->0 flips in n) needed.
        # Use the built-in method bin() to get the binary string representation
        # and then count the occurrences of '1'.
        count = bin(diff).count('1')

        return count