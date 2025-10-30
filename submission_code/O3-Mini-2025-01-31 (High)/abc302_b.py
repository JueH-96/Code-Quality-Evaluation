def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return

    # Read the grid dimensions
    H, W = map(int, data[0].split())
    grid = data[1:1+H]

    # The letter sequence we need to find
    target = "snuke"
    
    # All 8 possible directions (vertical, horizontal, and the two diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),   (1, 1)]

    # Iterate over each cell in the grid as a potential starting point
    for i in range(H):
        for j in range(W):
            if grid[i][j] != 's':
                continue  # The first letter must be 's'
            # Check each direction from this starting cell
            for dr, dc in directions:
                valid = True
                coords = []
                for k in range(5):
                    ni = i + dr * k
                    nj = j + dc * k
                    # Check boundaries
                    if ni < 0 or ni >= H or nj < 0 or nj >= W:
                        valid = False
                        break
                    # Check if the letter matches the target sequence
                    if grid[ni][nj] != target[k]:
                        valid = False
                        break
                    coords.append((ni, nj))
                if valid:
                    # Print the result converting to 1-indexed coordinates
                    for r, c in coords:
                        sys.stdout.write(f"{r+1} {c+1}
")
                    return

if __name__ == "__main__":
    main()