class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        """
        Calculates the number of ways to distribute n candies among 3 children
        such that no child gets more than limit candies.

        Args:
            n: The total number of candies (1 <= n <= 50).
            limit: The maximum number of candies any child can receive (1 <= limit <= 50).

        Returns:
            The total number of valid distribution ways.
        """
        # We are looking for the number of distinct non-negative integer tuples (c1, c2, c3)
        # such that c1 + c2 + c3 = n and 0 <= c1 <= limit, 0 <= c2 <= limit, 0 <= c3 <= limit.

        # Initialize a counter for the number of valid distribution ways.
        count = 0

        # We can iterate through all possible valid numbers of candies for the first two children,
        # and check if the remaining number of candies for the third child is also valid.

        # Iterate through all possible integer values for the number of candies
        # given to the first child (c1).
        # A child can receive 0 candies. The maximum number of candies a child can receive
        # is specified by 'limit'. So c1 can range from 0 to limit.
        for c1 in range(limit + 1):

            # Iterate through all possible integer values for the number of candies
            # given to the second child (c2).
            # Similar to c1, c2 can range from 0 to limit.
            for c2 in range(limit + 1):

                # The number of candies for the third child (c3) is determined by the
                # constraint that the total number of candies distributed must be exactly 'n'.
                # So, c3 = n - c1 - c2.

                # Now, we must check if this calculated value for c3 is valid according
                # to the problem constraints.
                # A valid number of candies for any child must be non-negative (>= 0)
                # and must not exceed the given limit (<= limit).
                if 0 <= c3 <= limit:
                    # If the number of candies for the third child (c3) is within the
                    # valid range [0, limit], then the triplet (c1, c2, c3) represents
                    # a valid way to distribute the 'n' candies according to all conditions.
                    # Increment the counter for valid distribution ways.
                    count += 1

        # After iterating through all possible combinations of (c1, c2) where c1 and c2
        # are within the range [0, limit], and checking the validity of the resulting c3,
        # the 'count' variable holds the total number of valid ways to distribute the candies.
        return count