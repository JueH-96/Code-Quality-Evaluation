class Solution:
    def maxPossibleScore(self, start: list[int], d: int) -> int:
        # Sort the start array to ensure non-decreasing order
        start.sort()
        
        # Initialize the maximum possible score to the maximum value of d
        max_score = d
        
        # Iterate through the intervals to find the minimum absolute difference
        for i in range(1, len(start)):
            # Calculate the minimum absolute difference between adjacent intervals
            min_diff = start[i] - (start[i - 1] + d)
            # Update the maximum possible score if a smaller difference is found
            max_score = min(max_score, min_diff)
        
        # Return the maximum possible score
        return max_score

# Example usage:
# sol = Solution()
# print(sol.maxPossibleScore([6,0,3], 2))  # Output: 4
# print(sol.maxPossibleScore([2,6,13,13], 5))  # Output: 5