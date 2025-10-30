class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        def solve(start_row, start_col, move_options):
            dp = {}

            def recurse(row, col, moves_left):
                if (row, col, moves_left) in dp:
                    return dp[(row, col, moves_left)]
                
                if moves_left == 0:
                    return fruits[row][col]
                
                max_fruits = 0
                for move_row, move_col in move_options(row, col):
                    if 0 <= move_row < n and 0 <= move_col < n:
                        max_fruits = max(max_fruits, fruits[row][col] + recurse(move_row, move_col, moves_left - 1))
                
                dp[(row, col, moves_left)] = max_fruits
                return max_fruits

            return recurse(start_row, start_col, n - 1)

        def move_options1(row, col):
            return [(row + 1, col + 1), (row + 1, col), (row, col + 1)]

        def move_options2(row, col):
            return [(row + 1, col - 1), (row + 1, col), (row + 1, col + 1)]

        def move_options3(row, col):
            return [(row - 1, col + 1), (row, col + 1), (row + 1, col + 1)]
        
        path1 = {}
        path2 = {}
        path3 = {}

        def solve_path(start_row, start_col, move_options, path):
            dp = {}

            def recurse(row, col, moves_left, current_path):
                if (row, col, moves_left) in dp:
                    return dp[(row, col, moves_left)]
                
                if moves_left == 0:
                    current_path.append((row, col))
                    return fruits[row][col]
                
                max_fruits = 0
                best_next_move = None
                for move_row, move_col in move_options(row, col):
                    if 0 <= move_row < n and 0 <= move_col < n:
                        fruits_collected = fruits[row][col] + recurse(move_row, move_col, moves_left - 1, current_path + [(row, col)])
                        if fruits_collected > max_fruits:
                            max_fruits = fruits_collected
                            best_next_move = (move_row, move_col)
                
                dp[(row, col, moves_left)] = max_fruits
                return max_fruits

            return recurse(start_row, start_col, n - 1, [])

        total_fruits = solve_path(0, 0, move_options1, path1) + solve_path(0, n - 1, move_options2, path2) + solve_path(n - 1, 0, move_options3, path3)
        
        visited = set()
        overlap = 0
        
        def get_path(start_row, start_col, move_options):
            path = [(start_row, start_col)]
            row, col = start_row, start_col
            for _ in range(n - 1):
                best_move = None
                max_val = -1
                for move_row, move_col in move_options(row, col):
                    if 0 <= move_row < n and 0 <= move_col < n:
                        if fruits[move_row][move_col] > max_val:
                            max_val = fruits[move_row][move_col]
                            best_move = (move_row, move_col)
                row, col = best_move
                path.append((row, col))
            return path
        
        path1 = get_path(0, 0, move_options1)
        path2 = get_path(0, n - 1, move_options2)
        path3 = get_path(n - 1, 0, move_options3)

        total_fruits = 0
        visited = set()
        
        for r, c in path1:
            if (r, c) not in visited:
                total_fruits += fruits[r][c]
                visited.add((r, c))
        for r, c in path2:
            if (r, c) not in visited:
                total_fruits += fruits[r][c]
                visited.add((r, c))
        for r, c in path3:
            if (r, c) not in visited:
                total_fruits += fruits[r][c]
                visited.add((r, c))
        
        return total_fruits