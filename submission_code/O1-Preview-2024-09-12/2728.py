class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        # Sort each row in descending order
        for row in nums:
            row.sort(reverse=True)
        # Find the maximum length among all rows
        max_length = max(len(row) for row in nums)
        # Iterate over each column index
        for i in range(max_length):
            max_element = 0
            # Collect the ith element from each row if it exists
            for row in nums:
                if i < len(row):
                    max_element = max(max_element, row[i])
            score += max_element
        return score