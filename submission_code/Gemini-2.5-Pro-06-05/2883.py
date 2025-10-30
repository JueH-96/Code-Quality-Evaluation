class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        from functools import lru_cache
        n = len(s)

        # Pre-compute the set of "beautiful" strings. A string is beautiful if it's
        # the binary representation of a power of 5 and has no leading zeros.
        # We only need to consider powers of 5 whose binary representation has a length
        # no greater than the input string's length `n`.
        beautiful_strings = set()
        p = 1
        while True:
            binary_representation = bin(p)[2:]
            if len(binary_representation) > n:
                break
            beautiful_strings.add(binary_representation)
            p *= 5

        # Use a memoized recursive helper function (Top-Down DP) to find the minimum partitions.
        @lru_cache(None)
        def _solve(start_index: int) -> int:
            """
            Returns the minimum number of beautiful substrings for the suffix s[start_index:].
            Returns float('inf') if no such partition exists.
            """
            # Base case: If we have successfully partitioned the entire string.
            if start_index == n:
                return 0

            # Optimization: A beautiful string must start with '1'.
            # If the current character is '0', we cannot start a valid substring here.
            if s[start_index] == '0':
                return float('inf')

            min_partitions = float('inf')

            # Iterate through all possible end points for the next substring.
            for end_index in range(start_index + 1, n + 1):
                substring = s[start_index:end_index]

                if substring in beautiful_strings:
                    # If the substring is beautiful, recursively solve for the rest of the string.
                    remaining_partitions = _solve(end_index)

                    # If the rest of the string can be partitioned, update the minimum count.
                    if remaining_partitions != float('inf'):
                        min_partitions = min(min_partitions, 1 + remaining_partitions)

            return min_partitions

        result = _solve(0)

        # If the result is infinity, it means no valid partition was found.
        return result if result != float('inf') else -1