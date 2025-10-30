class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict

        # Sort the coordinates by x and y
        coordinates.sort()

        # Create a dictionary to store the longest increasing path ending at each point
        dp = defaultdict(int)

        # Initialize the starting point
        start = tuple(coordinates[k])
        dp[start] = 1

        # Iterate through the sorted coordinates
        for x, y in coordinates:
            current = (x, y)
            if current != start:
                # Check all previous points that can be part of the increasing path
                for px, py in coordinates:
                    if px < x and py < y:
                        dp[current] = max(dp[current], dp[(px, py)] + 1)

        # The result is the longest path ending at the starting point
        return dp[start]