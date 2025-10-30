def main():
    import sys
    sys.setrecursionlimit(10000)
    
    N = int(sys.stdin.readline())
    R = sys.stdin.readline().strip()
    C = sys.stdin.readline().strip()
    
    grid = [['.' for _ in range(N)] for _ in range(N)]
    row_pos = [ -1 for _ in range(N)]  # row_pos[i] = j means R[i] is placed at (i,j)
    col_pos = [ -1 for _ in range(N)]  # col_pos[j] = i means C[j] is placed at (i,j)
    
    # Check if placing R[i] at (i,j) is valid
    def can_place_R(i, j):
        # All cells to the left of j in row i must be empty
        for jj in range(j):
            if grid[i][jj] != '.':
                return False
        return True
    
    # Check if placing C[j] at (i,j) is valid
    def can_place_C(j, i):
        # All cells above i in column j must be empty
        for ii in range(i):
            if grid[ii][j] != '.':
                return False
        return True
    
    # Check if the current grid is valid
    def is_valid():
        # Check rows
        for i in range(N):
            letters = set()
            for j in range(N):
                if grid[i][j] in {'A','B','C'}:
                    letters.add(grid[i][j])
            if len(letters) != 3:
                return False
        # Check columns
        for j in range(N):
            letters = set()
            for i in range(N):
                if grid[i][j] in {'A','B','C'}:
                    letters.add(grid[i][j])
            if len(letters) != 3:
                return False
        # Check row leftmost
        for i in range(N):
            for j in range(N):
                if grid[i][j] != '.':
                    if grid[i][j] != R[i]:
                        return False
                    break
        # Check column topmost
        for j in range(N):
            for i in range(N):
                if grid[i][j] != '.':
                    if grid[i][j] != C[j]:
                        return False
                    break
        return True
    
    # Backtracking function
    def backtrack(pos):
        if pos == N:
            # Now place C[j] positions
            for j in range(N):
                if col_pos[j] == -1:
                    for i in range(N):
                        if can_place_C(j, i) and grid[i][j] == '.':
                            grid[i][j] = C[j]
                            col_pos[j] = i
                            # Check if this placement conflicts with row's R
                            conflict = False
                            for jj in range(j):
                                if row_pos[i] == jj and grid[i][jj] != '.':
                                    if grid[i][jj] != C[j]:
                                        conflict = True
                                        break
                            if not conflict:
                                # Fill remaining cells
                                # Check rows
                                for row in range(N):
                                    placed = set()
                                    for col in range(N):
                                        if grid[row][col] in {'A','B','C'}:
                                            placed.add(grid[row][col])
                                    needed = {'A','B','C'} - placed
                                    # Find empty cells in the row and place the needed letters
                                    empty = [col for col in range(N) if grid[row][col] == '.']
                                    if len(empty) != len(needed):
                                        grid[i][j] = '.'
                                        col_pos[j] = -1
                                        continue
                                    # Assign letters to empty cells
                                    for col in empty:
                                        grid[row][col] = needed.pop()
                                # Check if the grid is valid
                                if is_valid():
                                    print("Yes")
                                    for row in grid:
                                        print(''.join(row))
                                    sys.exit(0)
                                # Backtrack
                                for row in range(N):
                                    for col in range(N):
                                        if grid[row][col] not in {'A','B','C'}:
                                            grid[row][col] = '.'
                            grid[i][j] = '.'
                            col_pos[j] = -1
                    return
            # All C[j] are placed, check validity
            if is_valid():
                print("Yes")
                for row in grid:
                    print(''.join(row))
                sys.exit(0)
            return
        # Place R[pos] in row pos
        for j in range(N):
            if can_place_R(pos, j):
                # Place R[pos] at (pos,j)
                grid[pos][j] = R[pos]
                row_pos[pos] = j
                # Ensure that for column j, if C[j] is to be placed at (pos,j), then C[j] == R[pos]
                if col_pos[j] != -1:
                    if col_pos[j] != pos or grid[pos][j] != C[j]:
                        grid[pos][j] = '.'
                        row_pos[pos] = -1
                        continue
                # Proceed to the next row
                backtrack(pos + 1)
                # Backtrack
                grid[pos][j] = '.'
                row_pos[pos] = -1
    
    backtrack(0)
    print("No")

if __name__ == "__main__":
    main()