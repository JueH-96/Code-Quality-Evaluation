class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in ascending order so that the largest element is at the end.
        for row in nums:
            row.sort()
        
        # Determine the maximum length among all rows.
        max_len = max(len(row) for row in nums)
        
        score = 0
        # For each "round" from the last column to the first (1 to max_len),
        # collect the i-th from the end for all rows having enough length.
        # Then take the maximum of those collected values and add to the score.
        for i in range(1, max_len + 1):
            candidates = []
            for row in nums:
                if len(row) >= i:
                    candidates.append(row[-i])
            if candidates:
                score += max(candidates)
        
        return score