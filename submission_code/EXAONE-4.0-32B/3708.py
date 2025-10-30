class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        sequence = []
        for i in range(len(grid)):
            if i % 2 == 0:
                for j in range(len(grid[i])):
                    sequence.append(grid[i][j])
            else:
                for j in range(len(grid[i])-1, -1, -1):
                    sequence.append(grid[i][j])
        return [sequence[i] for i in range(0, len(sequence), 2)]