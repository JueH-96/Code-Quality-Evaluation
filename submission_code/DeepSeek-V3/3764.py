class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        # Step 1: For each row, sort in descending order and take the top 'limits[i]' elements
        top_elements = []
        for i in range(len(grid)):
            row = grid[i]
            limit = limits[i]
            # Sort the row in descending order
            sorted_row = sorted(row, reverse=True)
            # Take up to 'limit' elements
            top_elements.extend(sorted_row[:limit])
        
        # Step 2: Sort all collected elements in descending order
        top_elements.sort(reverse=True)
        
        # Step 3: Take the top k elements (or all if there are less than k)
        max_sum = sum(top_elements[:k])
        
        return max_sum