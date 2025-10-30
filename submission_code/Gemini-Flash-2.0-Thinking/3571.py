class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)

        # dp_ending[i] stores the length of the longest increasing path ending at coordinates[i]
        dp_ending = [1] * n
        for i in range(n):
            for j in range(i):
                if coordinates[j][0] < coordinates[i][0] and coordinates[j][1] < coordinates[i][1]:
                    dp_ending[i] = max(dp_ending[i], 1 + dp_ending[j])

        # dp_starting[i] stores the length of the longest increasing path starting from coordinates[i]
        dp_starting = [1] * n
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if coordinates[i][0] < coordinates[j][0] and coordinates[i][1] < coordinates[j][1]:
                    dp_starting[i] = max(dp_starting[i], 1 + dp_starting[j])

        max_length = 0

        # Iterate through all possible increasing paths containing coordinates[k]
        # The path can be split into two parts: ending at coordinates[k] and starting from coordinates[k]

        # Length of the longest increasing path ending at coordinates[k]
        length_ending_at_k = dp_ending[k]

        # Length of the longest increasing path starting from coordinates[k]
        length_starting_from_k = dp_starting[k]

        max_length = length_ending_at_k + length_starting_from_k - 1

        return max_length