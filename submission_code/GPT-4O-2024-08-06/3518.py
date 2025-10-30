from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # Initialize variables
        n = len(b)
        max_score = float('-inf')
        
        # Iterate over all possible combinations of indices i_0, i_1, i_2, i_3
        for i0 in range(n - 3):
            for i1 in range(i0 + 1, n - 2):
                for i2 in range(i1 + 1, n - 1):
                    for i3 in range(i2 + 1, n):
                        # Calculate the score for the current combination
                        score = a[0] * b[i0] + a[1] * b[i1] + a[2] * b[i2] + a[3] * b[i3]
                        # Update the maximum score if the current score is greater
                        max_score = max(max_score, score)
        
        return max_score