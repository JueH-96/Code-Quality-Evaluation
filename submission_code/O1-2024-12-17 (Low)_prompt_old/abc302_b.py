def solve():
    import sys
    data = sys.stdin.read().strip().split()
    H, W = map(int, data[:2])
    grid = data[2:]
    
    # Define the sequence we look for
    target = "snuke"
    # All possible directions (8 directions: horizontal, vertical, diagonal)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for r in range(H):
        for c in range(W):
            if grid[r][c] == 's':
                # For each direction, attempt to match "snuke"
                for dr, dc in directions:
                    found = True
                    positions = []
                    for i in range(5):
                        rr = r + dr * i
                        cc = c + dc * i
                        # Check boundaries and letter
                        if not (0 <= rr < H and 0 <= cc < W):
                            found = False
                            break
                        if grid[rr][cc] != target[i]:
                            found = False
                            break
                        positions.append((rr + 1, cc + 1))  # switch to 1-based indexing
                    if found:
                        # We have found the unique sequence; print and return
                        for pos in positions:
                            print(pos[0], pos[1])
                        return

def main():
    solve()

if __name__ == "__main__":
    main()