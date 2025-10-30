class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        from collections import defaultdict

        # Collect all unique values across the grid
        unique_vals = set()
        for row in grid:
            for val in row:
                unique_vals.add(val)
        unique_vals = list(unique_vals)
        val_to_bit = {v: i for i, v in enumerate(unique_vals)}
        num_unique = len(unique_vals)

        # Preprocess each row to have its unique values
        rows_unique = []
        for row in grid:
            current = set()
            for val in row:
                current.add(val)
            rows_unique.append(list(current))

        # Initialize DP with mask 0 (no values selected) and sum 0
        dp = defaultdict(int)
        dp[0] = 0

        # Process each row to update the DP states
        for row in rows_unique:
            new_dp = defaultdict(int)
            # Iterate over all existing masks and their sums
            for mask in dp:
                current_sum = dp[mask]
                # Option 1: Skip the current row
                if new_dp[mask] < current_sum:
                    new_dp[mask] = current_sum
                # Option 2: Try each unique value in the current row
                for val in row:
                    bit = val_to_bit[val]
                    if not (mask & (1 << bit)):
                        new_mask = mask | (1 << bit)
                        new_sum = current_sum + val
                        if new_sum > new_dp[new_mask]:
                            new_dp[new_mask] = new_sum
            # Update dp to be new_dp for processing the next row
            dp = new_dp

        # Find the maximum score excluding the mask 0 (no selections)
        max_score = 0
        for mask, score in dp.items():
            if mask != 0 and score > max_score:
                max_score = score
        return max_score