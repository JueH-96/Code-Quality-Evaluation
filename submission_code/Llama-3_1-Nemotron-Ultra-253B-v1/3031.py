from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        if not grid:
            return []
        n = len(grid)
        m = len(grid[0])
        flat = [num for row in grid for num in row]
        length = len(flat)
        prefix = [1] * length
        for i in range(1, length):
            prefix[i] = (prefix[i-1] * flat[i-1]) % 12345
        suffix = [1] * length
        for i in range(length-2, -1, -1):
            suffix[i] = (suffix[i+1] * flat[i+1]) % 12345
        result_flat = [(prefix[i] * suffix[i]) % 12345 for i in range(length)]
        result = []
        for i in range(n):
            start = i * m
            result.append(result_flat[start:start+m])
        return result