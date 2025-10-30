class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7

        # Helper function to calculate the number of ways to arrange the sequence
        def count_ways(left, right):
            if left == right:
                return 1
            if left > right:
                return 0

            # Calculate the number of ways to arrange the sequence
            ways = 1
            for i in range(left, right):
                ways = (ways * (i - left + 1)) % MOD
            return ways

        # Initialize the result
        result = 1

        # Iterate through the sick array and calculate the number of ways
        for i in range(len(sick) - 1):
            left = sick[i] + 1
            right = sick[i + 1] - 1
            result = (result * count_ways(left, right)) % MOD

        # Calculate the number of ways for the leftmost and rightmost segments
        result = (result * count_ways(0, sick[0] - 1)) % MOD
        result = (result * count_ways(sick[-1] + 1, n - 1)) % MOD

        return result