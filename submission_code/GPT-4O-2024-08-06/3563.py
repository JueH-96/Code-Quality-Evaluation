class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        from itertools import permutations
        
        def is_unique_and_valid(selection):
            # Check if all values in the selection are unique
            return len(selection) == len(set(selection))
        
        max_score = 0
        rows = len(grid)
        
        # Generate all permutations of column indices for the number of rows
        for perm in permutations(range(len(grid[0])), rows):
            # Select one element from each row based on the current permutation
            selection = [grid[i][perm[i]] for i in range(rows)]
            
            # Check if the selection is valid (unique values)
            if is_unique_and_valid(selection):
                # Calculate the score and update max_score if it's higher
                score = sum(selection)
                max_score = max(max_score, score)
        
        return max_score