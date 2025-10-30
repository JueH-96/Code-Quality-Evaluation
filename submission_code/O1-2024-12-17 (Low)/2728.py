class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in ascending order
        for row in nums:
            row.sort()
        
        # Find the maximum length among all rows
        max_len = max(len(row) for row in nums)
        
        score = 0
        # Iterate from 1 to max_len
        for i in range(1, max_len + 1):
            removed_values = []
            # Collect the i-th largest element from each row (if it exists)
            for row in nums:
                if len(row) >= i:
                    removed_values.append(row[-i])
            # The highest number among the removed values is added to the score
            score += max(removed_values)
        
        return score