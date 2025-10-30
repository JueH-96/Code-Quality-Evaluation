from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        # First, build the complete zigzag traversal order
        order = []
        for i, row in enumerate(grid):
            # For even row index: traverse left to right; odd: traverse right to left.
            if i % 2 == 0:
                order.extend(row)
            else:
                order.extend(row[::-1])
        
        # Now, skip every alternate cell: take the 0th, 2nd, 4th, ...
        result = [order[idx] for idx in range(0, len(order), 2)]
        return result

# Example test cases to validate the solution:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    grid1 = [[1,2],[3,4]]
    print(sol.zigzagTraversal(grid1))  # Expected: [1,4]
    
    # Example 2:
    grid2 = [[2,1],[2,1],[2,1]]
    print(sol.zigzagTraversal(grid2))  # Expected: [2,1,2]
    
    # Example 3:
    grid3 = [[1,2,3],[4,5,6],[7,8,9]]
    print(sol.zigzagTraversal(grid3))  # Expected: [1,3,5,7,9]