import sys

# Setting a higher recursion depth is generally not needed for iterative solutions
# but good practice in competitive programming if recursion is deep.
# sys.setrecursionlimit(2000) 

class Solution:
    """
    Solves the problem of finding the maximum number of operations on a binary string.
    An operation consists of choosing s[i]='1', s[i+1]='0' and moving s[i] right
    past subsequent '0's until it hits end of string or another '1'.
    """
    def maxOperations(self, s: str) -> int:
        """
        Calculates the maximum number of operations that can be performed using a linear scan.
        
        The core idea is to count the contributions to the total operations as we scan the string.
        An operation involves a '1' moving rightwards over '0's. This happens when a '1' is immediately followed by a '0'.
        The total number of operations seems to be related to the cumulative count of '1's when such transitions occur.

        Consider scanning the string from left to right. Maintain a count of '1's seen so far (`ones_count`).
        When a '1' is encountered, increment `ones_count`.
        When a '0' is encountered:
        If there were '1's preceding this '0' (`ones_count > 0`), this '0' represents an obstacle that the '1's must eventually pass.
        Specifically, if this '0' is the *first* '0' immediately following a block of one or more '1's, it triggers a set of potential moves.
        The number of operations associated with this trigger event appears to be equal to the total number of '1's encountered so far (`ones_count`).
        We use a flag (`found_zero_after_one`) to ensure we only add this contribution once for the first '0' after a block of '1's, even if there are multiple consecutive '0's.

        Example trace `s = "1001101"`:
        - `i=0, s[0]='1'`: `ones_count = 1`. `found_zero_after_one = False`. `res = 0`.
        - `i=1, s[1]='0'`: `ones_count = 1 > 0`. `found_zero_after_one` is False. Add `ones_count` (1) to `res`. `res = 1`. Set `found_zero_after_one = True`.
        - `i=2, s[2]='0'`: `ones_count = 1 > 0`. `found_zero_after_one` is True. Do nothing.
        - `i=3, s[3]='1'`: `ones_count = 2`. `found_zero_after_one = False`.
        - `i=4, s[4]='1'`: `ones_count = 3`. `found_zero_after_one = False`.
        - `i=5, s[5]='0'`: `ones_count = 3 > 0`. `found_zero_after_one` is False. Add `ones_count` (3) to `res`. `res = 1 + 3 = 4`. Set `found_zero_after_one = True`.
        - `i=6, s[6]='1'`: `ones_count = 4`. `found_zero_after_one = False`.
        Final `res = 4`. Matches Example 1.

        Args:
            s: The input binary string.

        Returns:
            The maximum number of operations possible.
        """
        n = len(s)
        # Stores the total count of operations.
        res = 0
        # Stores the count of '1's encountered so far.
        ones_count = 0
        # Flag to track if the current sequence of characters includes a '0' that immediately followed a block of '1's.
        # This helps to ensure that we count operations only for the start of a block of '0's following '1's.
        found_zero_after_one = False 
        
        for i in range(n):
            if s[i] == '1':
                # If the character is '1', increment the count of '1's.
                ones_count += 1
                # Reset the flag because we have encountered a '1'.
                # Any subsequent '0' will mark the start of a new potential move sequence trigger point.
                found_zero_after_one = False 
            else: # s[i] == '0'
                # If the character is '0', check if there are accumulated '1's to its left (`ones_count > 0`)
                # and if this is the first '0' encountered right after the last '1' (or block of '1's).
                # The flag `found_zero_after_one` helps track this condition.
                if ones_count > 0 and not found_zero_after_one:
                    # If both conditions are true, this '0' signifies a point where operations
                    # involving the accumulated '1's are required. Add the current count of '1's
                    # to the total result. This represents the sum of operations performed by
                    # each '1' encountered so far crossing over this boundary block start.
                    res += ones_count
                    # Set the flag to True. This prevents adding `ones_count` again for
                    # subsequent consecutive '0's within the same block, as they are part
                    # of the same barrier crossing event triggered by the first '0'.
                    found_zero_after_one = True
                    
        return res