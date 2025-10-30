class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Step 1: Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        
        # Step 2: Find the maximum row length
        max_length = max(len(row) for row in nums)
        
        # Step 3: Initialize score
        score = 0
        
        # Step 4: Perform operations
        for j in range(max_length):
            max_val = -1  # Initialize max_val for this operation
            for row in nums:
                if j < len(row):
                    current_val = row[j]
                    if current_val > max_val:
                        max_val = current_val
            # Add the maximum value of this operation to the score
            score += max_val
        
        # Step 5: Return the final score
        return score