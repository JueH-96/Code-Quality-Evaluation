class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        def check_square(r, c):
            colors = {grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]}
            return len(colors) == 1
        
        for r in [0, 1]:
            for c in [0, 1]:
                if check_square(r, c):
                    return True
        
        for i in range(3):
            for j in range(3):
                original = grid[i][j]
                new_char = 'W' if original == 'B' else 'B'
                
                squares_to_check = []
                for r0 in [0, 1]:
                    for c0 in [0, 1]:
                        if r0 <= i <= r0 + 1 and c0 <= j <= c0 + 1:
                            squares_to_check.append((r0, c0))
                
                for (r0, c0) in squares_to_check:
                    colors_set = set()
                    for dr in [0, 1]:
                        for dc in [0, 1]:
                            ni, nj = r0 + dr, c0 + dc
                            if ni == i and nj == j:
                                colors_set.add(new_char)
                            else:
                                colors_set.add(grid[ni][nj])
                    if len(colors_set) == 1:
                        return True
        
        return False