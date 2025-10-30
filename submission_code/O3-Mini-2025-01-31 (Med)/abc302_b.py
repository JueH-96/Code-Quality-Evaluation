def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    H = int(data[0])
    W = int(data[1])
    grid = data[2:]
    
    # The target word we are looking for
    target = "snuke"
    
    # All possible directions: vertical, horizontal, and diagonal (8 directions)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Iterate over every cell as potential starting cell for 's'
    for i in range(H):
        for j in range(W):
            if grid[i][j] != 's':
                continue  # early skip if starting letter isn't 's'
            # Try each direction from the starting cell
            for dr, dc in directions:
                valid = True
                coords = []  # to store the 5 positions (row, col)
                for step in range(5):
                    r = i + dr * step
                    c = j + dc * step
                    # Check bounds
                    if 0 <= r < H and 0 <= c < W:
                        # Check if the character matches the target letter at this position
                        if grid[r][c] != target[step]:
                            valid = False
                            break
                        # Save coordinates as 1-indexed
                        coords.append((r + 1, c + 1))
                    else:
                        valid = False
                        break
                if valid:
                    # Print the valid sequence in the required format
                    output = "
".join(f"{r} {c}" for r, c in coords)
                    sys.stdout.write(output)
                    return

if __name__ == '__main__':
    main()