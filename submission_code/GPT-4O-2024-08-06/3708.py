class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        skip = False  # This will help us skip every alternate cell
        for i, row in enumerate(grid):
            if i % 2 == 0:
                # Traverse from left to right
                for j in range(len(row)):
                    if not skip:
                        result.append(row[j])
                    skip = not skip
            else:
                # Traverse from right to left
                for j in range(len(row) - 1, -1, -1):
                    if not skip:
                        result.append(row[j])
                    skip = not skip
        return result