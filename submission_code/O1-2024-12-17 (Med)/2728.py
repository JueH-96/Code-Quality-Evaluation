class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in ascending order
        for row in nums:
            row.sort()

        score = 0
        # Find the maximum length among all rows
        max_len = max(len(row) for row in nums)

        # For each position from the end (0 to max_len-1), pick the element
        # from each row if it exists, and add the largest of these picks to score.
        for i in range(max_len):
            candidate_values = []
            for row in nums:
                if len(row) > i:
                    candidate_values.append(row[-1 - i])
            score += max(candidate_values)

        return score