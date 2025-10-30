def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, N = map(int, input_data[:3])
    T = input_data[3]
    S = input_data[4:]

    # Precompute which cells are land as bitmasks per row.
    # For row r, the c-th bit (0 <= c < W) is set to 1 if (r,c) is land ('.').
    # c=0 corresponds to the least significant bit.
    land = [0]*H
    for r in range(H):
        row_mask = 0
        for c in range(W):
            if S[r][c] == '.':
                row_mask |= (1 << c)
        land[r] = row_mask

    # Current possible positions (as a bitmask per row).
    # Initially, any land cell can be a starting position.
    curr = land[:]

    # A mask to avoid shifting bits beyond column W.
    mask = (1 << W) - 1

    for move in T:
        new_state = [0]*H

        if move == 'L':
            # Moving left: final col c comes from col c+1
            # So "shift right" by 1 in the bitmask, then AND with land to ensure valid land cells.
            for r in range(H):
                new_state[r] = (curr[r] >> 1) & land[r]

        elif move == 'R':
            # Moving right: final col c comes from col c-1
            # So "shift left" by 1 in the bitmask, mask out overflow, then AND with land.
            for r in range(H):
                new_state[r] = ((curr[r] << 1) & mask) & land[r]

        elif move == 'U':
            # Moving up: row r comes from row r+1
            # r=H-1 cannot come from H (out of grid), so we set new_state[H-1] = 0
            for r in range(H-1):
                new_state[r] = curr[r+1] & land[r]
            new_state[H-1] = 0

        else:  # move == 'D'
            # Moving down: row r comes from row r-1
            # r=0 cannot come from -1 (out of grid), so we set new_state[0] = 0
            for r in range(1, H):
                new_state[r] = curr[r-1] & land[r]
            new_state[0] = 0

        curr = new_state

    # Count how many bits are set across all rows in curr.
    # (Requires Python 3.10+ for .bit_count(). If not available, use bin(x).count('1').)
    answer = sum(row.bit_count() for row in curr)
    print(answer)

# Do not forget to call main!
main()