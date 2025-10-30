class Solution:
    def maxOperations(self, s: str) -> int:
        """
        We want to count the maximum number of times we can perform the given "move a '1' right past
        consecutive zeros until it meets the next '1' or the end" operation.  Crucially, each such
        move requires that we have a consecutive '10' somewhere, and it moves that '1' over all
        consecutive zeros in one go, but it may do so multiple times if there are further zeros
        later (once intervening '1's have moved).

        A cleaner way to see it is that each '1' can eventually cross every block of consecutive zeros
        that lies to its right (provided no entrenched '1' is still between them in the end, but
        eventually all '1's shuffle right).  Each crossing corresponds to exactly one operation.

        Concretely, if we "group" consecutive zeros into blocks, then for each '1' at index i,
        every zero block whose start index is strictly greater than i can be crossed by that '1'
        (one operation per block).  Summing this over all '1's gives the maximum number of operations.

        Example: s = "1001101"

        - Zero blocks start at indices [1, 5].
        - The '1's are at indices [0, 3, 4, 6].
        - For the '1' at index 0, both zero blocks start > 0 → it can eventually cross two blocks.
        - For '1' at index 3, only block at index 5 starts > 3 → crosses one block.
        - For '1' at index 4, only block at index 5 starts > 4 → crosses one block.
        - For '1' at index 6, no block starts > 6 → crosses zero blocks.
        Total = 2 + 1 + 1 + 0 = 4.

        We'll implement this as follows:
          1) Identify the starting index of each block of consecutive zeros in zero_starts.
          2) Collect all indices i for which s[i] == '1'.
          3) For each such i, do a binary search in zero_starts to count how many blocks start > i.
             Add that count to the answer.

        This correctly yields the maximum number of operations.
        """

        import bisect

        n = len(s)
        zero_starts = []
        # Find the start index of each block of consecutive zeros
        for i in range(n):
            if s[i] == '0':
                # If it's the first character or the previous char is '1',
                # then we've found the start of a zero-block.
                if i == 0 or s[i - 1] == '1':
                    zero_starts.append(i)

        # Collect indices of '1'
        ones = [i for i, ch in enumerate(s) if ch == '1']

        # For each '1', count how many zero-blocks start strictly after i
        # i.e. zero_starts[j] > i
        # This can be done with bisect_left(zero_starts, i+1)
        # Number of such blocks = len(zero_starts) - index
        ans = 0
        for i in ones:
            idx = bisect.bisect_left(zero_starts, i + 1)
            ans += len(zero_starts) - idx

        return ans