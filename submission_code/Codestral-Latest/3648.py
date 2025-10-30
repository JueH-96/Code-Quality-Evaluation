class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)

        # Directions for the three children
        directions = [
            [(1, 0), (0, 1), (1, 1)],  # Child 1
            [(1, 0), (0, -1), (1, -1)],  # Child 2
            [(0, 1), (-1, 1), (1, 1)]  # Child 3
        ]

        # Starting positions for the three children
        starts = [(0, 0), (0, n - 1), (n - 1, 0)]

        # Function to check if a move is within bounds
        def in_bounds(x, y):
            return 0 <= x < n and 0 <= y < n

        # Function to perform DFS and collect fruits
        def dfs(x, y, child_index, visited):
            if (x, y) in visited:
                return 0
            visited.add((x, y))
            max_fruits = fruits[x][y]
            for dx, dy in directions[child_index]:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny):
                    max_fruits += dfs(nx, ny, child_index, visited)
            return max_fruits

        # Initialize visited sets for each child
        visited1 = set()
        visited2 = set()
        visited3 = set()

        # Collect fruits for each child
        fruits1 = dfs(*starts[0], 0, visited1)
        fruits2 = dfs(*starts[1], 1, visited2)
        fruits3 = dfs(*starts[2], 2, visited3)

        # Total fruits collected
        total_fruits = fruits1 + fruits2 + fruits3

        return total_fruits