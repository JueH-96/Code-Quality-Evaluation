class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0

        # Sort each row in ascending order
        for row in nums:
            row.sort()

        while any(nums):  # Continue as long as there are elements in the matrix
            removed_elements = []
            for row in nums:
                if row:
                    removed_elements.append(row.pop())

            if removed_elements:
                score += max(removed_elements)

        return score