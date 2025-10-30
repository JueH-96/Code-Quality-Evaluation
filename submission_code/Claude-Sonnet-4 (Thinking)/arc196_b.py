def solve():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(input().strip())
    
    MOD = 998244353
    
    # Orientations: (top, right, bottom, left) - whether each edge has an endpoint
    orientations = {
        'A': [
            (1, 1, 0, 0),  # top-right
            (0, 1, 1, 0),  # right-bottom  
            (0, 0, 1, 1),  # bottom-left
            (1, 0, 0, 1)   # left-top
        ],
        'B': [
            (1, 0, 1, 0),  # top-bottom
            (0, 1, 0, 1)   # left-right
        ]
    }
    
    count = 0
    assignment = [[None] * W for _ in range(H)]
    
    def backtrack(pos):
        nonlocal count
        
        if pos == H * W:
            count += 1
            return
        
        i, j = divmod(pos, W)
        tile_type = grid[i][j]
        
        for top, right, bottom, left in orientations[tile_type]:
            valid = True
            
            # Check constraint with left neighbor
            if j > 0:
                prev_right = assignment[i][j-1][1]
                if left != prev_right:
                    valid = False
            
            # Check constraint with top neighbor  
            if i > 0:
                prev_bottom = assignment[i-1][j][2]
                if top != prev_bottom:
                    valid = False
            
            # Check wrap-around constraint for rightmost column
            if valid and j == W - 1:
                first_left = assignment[i][0][3]
                if right != first_left:
                    valid = False
            
            # Check wrap-around constraint for bottom row
            if valid and i == H - 1:
                first_top = assignment[0][j][0]
                if bottom != first_top:
                    valid = False
            
            if valid:
                assignment[i][j] = (top, right, bottom, left)
                backtrack(pos + 1)
    
    backtrack(0)
    return count % MOD

def main():
    T = int(input())
    for _ in range(T):
        print(solve())

if __name__ == "__main__":
    main()