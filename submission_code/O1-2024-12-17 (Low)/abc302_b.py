def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid = input_data[2:]

    # The word we are looking for:
    target = "snuke"
    # All possible directions (vertical, horizontal, diagonal)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    # Check each cell as a starting point if it is 's'
    for r in range(H):
        for c in range(W):
            if grid[r][c] == target[0]:  # 's'
                # Try all directions
                for dr, dc in directions:
                    # Collect candidate cells
                    cells = [(r, c)]
                    valid = True
                    # Move along the direction for the next 4 letters
                    for i in range(1, len(target)):
                        nr = r + dr*i
                        nc = c + dc*i
                        # Check boundary
                        if not(0 <= nr < H and 0 <= nc < W):
                            valid = False
                            break
                        # Check letter
                        if grid[nr][nc] != target[i]:
                            valid = False
                            break
                        cells.append((nr, nc))
                    # If valid and we have 5 cells, print and return
                    if valid and len(cells) == 5:
                        for (row_out, col_out) in cells:
                            print(row_out+1, col_out+1)
                        return

# Call main() to run the solution
if __name__ == "__main__":
    main()