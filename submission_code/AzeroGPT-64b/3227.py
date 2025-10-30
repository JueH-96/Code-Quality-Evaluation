class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        from collections import Counter
        flatten = [num for line in grid for num in line]
        counter = Counter(flatten)
        for i in range(1, (len(grid)**2)+1):
            if counter[i] == 2:
                repeat = i
            elif counter[i] == 0:
                missing = i
        return [repeat, missing]