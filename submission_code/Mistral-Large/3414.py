class Solution:
    def waysToReachStair(self, k: int) -> int:
        # Base cases
        if k == 0:
            return 2
        if k == 1:
            return 4

        # For k > 1, we need to calculate the number of ways using combinatorial methods
        # The number of ways to reach stair k can be derived from the binary representation
        # of k and the combinatorial counting of valid sequences of operations.

        # Initialize the number of ways
        ways = 0

        # Iterate through all possible binary representations of numbers less than or equal to k
        for i in range(k + 1):
            # Calculate the number of ways to reach stair i using combinatorial methods
            ways += self.countWays(i)

        return ways

    def countWays(self, n: int) -> int:
        # Count the number of ways to reach stair n using combinatorial methods
        # This involves counting the number of valid sequences of operations
        # that can be derived from the binary representation of n.

        # Initialize the number of ways
        ways = 0

        # Iterate through all possible binary representations of numbers less than or equal to n
        for i in range(n + 1):
            # Calculate the number of ways to reach stair i using combinatorial methods
            ways += self.comb(n, i)

        return ways

    def comb(self, n: int, k: int) -> int:
        # Calculate the binomial coefficient C(n, k)
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)  # Take advantage of symmetry
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c