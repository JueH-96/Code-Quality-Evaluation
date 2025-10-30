class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        nums = [num for row in grid for num in row]
        counts = [0] * (n * n + 1)
        for num in nums:
            counts[num] += 1
        a = b = None
        for i in range(1, n * n + 1):
            if counts[i] == 0:
                b = i
            elif counts[i] == 2:
                a = i
            if a is not None and b is not None:
                break
        return [a, b]