from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        # Create a list of tuples (value, index) and sort it by value
        indexed_b = sorted((val, idx) for idx, val in enumerate(b))

        # Create a prefix sum array for the sorted values
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + indexed_b[i - 1][0]

        max_score = float('-inf')

        # Iterate through possible starting points for the 4 indices
        for i in range(n - 3):
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n - 1):
                    for l in range(k + 1, n):
                        # Calculate the score using the prefix sums
                        score = (a[0] * (prefix_sum[j] - prefix_sum[i]) +
                                 a[1] * (prefix_sum[k] - prefix_sum[j]) +
                                 a[2] * (prefix_sum[l] - prefix_sum[k]) +
                                 a[3] * (prefix_sum[n] - prefix_sum[l]))
                        max_score = max(max_score, score)

        return max_score