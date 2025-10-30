class Solution:
    def getSmallestString(self, s: str) -> str:
        n = len(s)
        s_list = list(s) # Convert to list for mutable operations

        for i in range(n - 1):
            d1 = int(s_list[i])
            d2 = int(s_list[i+1])

            # Check if they have the same parity
            # (d1 % 2) == 0 means d1 is even
            # (d1 % 2) == 1 means d1 is odd
            # Same parity means both are even or both are odd
            if (d1 % 2) == (d2 % 2):
                # Check if swapping would make it lexicographically smaller
                # This happens if the digit on the right is smaller than the digit on the left
                if d2 < d1:
                    # Perform the swap
                    s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                    # The problem asks for the lexicographically smallest string
                    # achievable with AT MOST one swap. Iterating from left to right,
                    # the first time we find a valid swap (adjacent digits of same parity
                    # where the right is smaller than the left), performing that swap
                    # yields the lexicographically smallest result among all possible
                    # single swaps. Subsequent swaps at later indices would only affect
                    # digits to the right of the current position i+1, while this swap
                    # improves the string at position i, which has a higher impact
                    # lexicographically. Therefore, we perform the first such swap found
                    # and return the result.
                    return "".join(s_list)

        # If the loop finishes without finding any adjacent pair that can be swapped
        # to make the string lexicographically smaller, the original string is
        # already the smallest possible with at most one valid swap.
        return s