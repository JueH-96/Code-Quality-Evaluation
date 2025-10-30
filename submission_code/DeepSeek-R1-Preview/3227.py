class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        # Flatten the grid into a single list
        flattened = []
        for row in grid:
            flattened.extend(row)
        
        # Count the frequency of each number
        count = {}
        for num in flattened:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
        
        # Find the repeating number 'a'
        a = None
        for key, value in count.items():
            if value == 2:
                a = key
                break  # Found the repeating number
        
        # Calculate the expected sum of numbers from 1 to n^2
        n = len(grid)
        n_squared = n * n
        S = n_squared * (n_squared + 1) // 2
        
        # Calculate the actual sum of the grid elements
        sum_grid = sum(flattened)
        
        # Compute the missing number 'b'
        b = a + S - sum_grid
        
        return [a, b]