def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1
    N = int(input[ptr])
    ptr += 1
    T = input[ptr]
    ptr += 1

    # Precompute the moves as (di, dj)
    moves = []
    for c in T:
        if c == 'L':
            moves.append((0, -1))
        elif c == 'R':
            moves.append((0, 1))
        elif c == 'U':
            moves.append((-1, 0))
        else:  # 'D'
            moves.append((1, 0))

    grid = []
    for _ in range(H):
        grid.append(input[ptr])
        ptr += 1

    final_positions = set()

    for i in range(2, H):  # 1-based i from 2 to H-1 inclusive
        for j in range(2, W):  # 1-based j from 2 to W-1 inclusive
            if grid[i-1][j-1] == '.':
                current_i = i
                current_j = j
                valid = True
                for di, dj in moves:
                    new_i = current_i + di
                    new_j = current_j + dj
                    if not (1 <= new_i <= H and 1 <= new_j <= W):
                        valid = False
                        break
                    if grid[new_i - 1][new_j - 1] != '.':
                        valid = False
                        break
                    current_i = new_i
                    current_j = new_j
                if valid:
                    final_positions.add((current_i, current_j))

    print(len(final_positions))

if __name__ == "__main__":
    main()