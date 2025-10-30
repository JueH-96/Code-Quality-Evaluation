def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, N = map(int, input_data[:3])
    T = input_data[3]
    S = input_data[4:]
    
    # Directions map
    moves = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    
    # Read grid to identify land/sea
    # Land = True, Sea = False
    land = [[(S[r][c] == '.') for c in range(W)] for r in range(H)]
    
    # A 2D boolean array storing if a position can be Takahashi's position.
    # Initially, any land cell can be a candidate.
    possible = [[land[r][c] for c in range(W)] for r in range(H)]
    
    for move in T:
        dr, dc = moves[move]
        # Prepare the next state of possible positions
        next_possible = [[False]*W for _ in range(H)]
        # Shift all current possible positions by the move
        for r in range(H):
            row_possible = possible[r]
            for c in range(W):
                if row_possible[c]:
                    nr = r + dr
                    nc = c + dc
                    # Check if in bounds and is land (though perimeter is always sea)
                    if 0 <= nr < H and 0 <= nc < W and land[nr][nc]:
                        next_possible[nr][nc] = True
        possible = next_possible
    
    # Count how many positions remain possible
    ans = 0
    for r in range(H):
        ans += sum(possible[r])
    
    print(ans)

# Call main() to execute solution
if __name__ == "__main__":
    main()