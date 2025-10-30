def rotate_90(grid):
    N = len(grid)
    return [[grid[N-1-j][i] for j in range(N)] for i in range(N)]

def get_piece_coords(grid):
    coords = []
    for i in range(4):
        for j in range(4):
            if grid[i][j] == '#':
                coords.append((i,j))
    return coords

def normalize_coords(coords):
    if not coords:
        return coords
    min_x = min(x for x,y in coords)
    min_y = min(y for x,y in coords)
    return [(x-min_x, y-min_y) for x,y in coords]

def get_all_rotations(piece):
    rotations = []
    curr = piece
    for _ in range(4):
        norm = normalize_coords(get_piece_coords(curr))
        if norm not in rotations:
            rotations.append(norm)
        curr = rotate_90(curr)
    return rotations

def can_place(board, piece, x, y):
    for px, py in piece:
        nx, ny = x + px, y + py
        if nx < 0 or nx >= 4 or ny < 0 or ny >= 4 or board[nx][ny]:
            return False
    return True

def place_piece(board, piece, x, y, val):
    for px, py in piece:
        board[x+px][y+py] = val

def solve(board, pieces, used, pos=0):
    if pos >= 16:
        return all(all(cell for cell in row) for row in board)
    
    x, y = pos // 4, pos % 4
    if board[x][y]:
        return solve(board, pieces, used, pos+1)
        
    for i, rotations in enumerate(pieces):
        if used[i]:
            continue
        for piece in rotations:
            if can_place(board, piece, x, y):
                used[i] = True
                place_piece(board, piece, x, y, True)
                if solve(board, pieces, used, pos+1):
                    return True
                place_piece(board, piece, x, y, False)
                used[i] = False
    return False

def main():
    pieces = []
    for _ in range(3):
        grid = []
        for _ in range(4):
            row = list(input().strip())
            grid.append(row)
        pieces.append(get_all_rotations(grid))
    
    board = [[False]*4 for _ in range(4)]
    used = [False]*3
    
    print("Yes" if solve(board, pieces, used) else "No")

main()