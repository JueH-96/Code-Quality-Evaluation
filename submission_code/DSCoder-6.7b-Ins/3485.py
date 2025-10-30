class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        # Sort the start array
        start.sort()

        # Initialize the maximum possible score as 0
        max_score = 0

        # Iterate over the start array
        for i in range(len(start) - 1):
            # Calculate the score for the current pair of integers
            score = min(abs(start[i] - start[i+1]), abs(start[i] + d - start[i+1]), abs(start[i] - start[i+1] - d))

            # Update the maximum possible score
            max_score = max(max_score, score)

        # Return the maximum possible score
        return max_score