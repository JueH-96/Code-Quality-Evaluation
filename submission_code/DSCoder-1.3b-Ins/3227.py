class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        num_set = set()
        repeated = []
        
        for row in grid:
            for num in row:
                if num in num_set:
                    repeated.append(num)
                else:
                    num_set.add(num)
        
        missing = [i for i in range(1, n*n+1) if i not in num_set]
        
        return [repeated[0], missing[0]]