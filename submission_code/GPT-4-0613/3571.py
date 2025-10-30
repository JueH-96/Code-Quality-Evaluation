class Solution:
    def maxPathLength(self, coordinates, k):
        coordinates.sort()
        dp = [1] * len(coordinates)
        coordinates_dict = {tuple(coordinate): i for i, coordinate in enumerate(coordinates)}
        for i in range(len(coordinates)):
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = coordinates[i][0] + dx, coordinates[i][1] + dy
                if (nx, ny) in coordinates_dict and coordinates_dict[(nx, ny)] < i:
                    dp[i] = max(dp[i], dp[coordinates_dict[(nx, ny)]] + 1)
        return dp[coordinates_dict[tuple(coordinates[k])]]