from itertools import permutations

def solve():
    H, W = map(int, input().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, input().split())))
    B = []
    for _ in range(H):
        B.append(list(map(int, input().split())))

    def calculate_moves(initial_grid, target_grid):
        min_moves = float('inf')
        
        row_perms = list(permutations(range(H)))
        col_perms = list(permutations(range(W)))
        
        for row_perm in row_perms:
            for col_perm in col_perms:
                
                temp_grid = []
                for i in row_perm:
                    row = []
                    for j in col_perm:
                        row.append(initial_grid[i][j])
                    temp_grid.append(row)
                
                if temp_grid == target_grid:
                    moves = 0
                    
                    # Calculate row moves
                    row_order = list(row_perm)
                    row_moves = 0
                    for i in range(H):
                        for j in range(i + 1, H):
                            if row_order[i] > row_order[j]:
                                row_moves += 1
                    
                    # Calculate column moves
                    col_order = list(col_perm)
                    col_moves = 0
                    for i in range(W):
                        for j in range(i + 1, W):
                            if col_order[i] > col_order[j]:
                                col_moves += 1
                    
                    moves = row_moves + col_moves
                    min_moves = min(min_moves, moves)
        
        if min_moves == float('inf'):
            return -1
        else:
            return min_moves

    print(calculate_moves(A, B))

solve()