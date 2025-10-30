import sys
from collections import defaultdict

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    S = [list(sys.stdin.readline().strip()) for _ in range(H)]

    # Transpose the grid to minimize the row width (W)
    if H > W:
        transposed = []
        for j in range(W):
            new_row = []
            for i in range(H):
                new_row.append(S[i][j])
            transposed.append(new_row)
        S = transposed
        H, W = W, H

    valid_masks = []
    for row in S:
        masks = []
        # Generate all valid masks for the current row using backtracking
        def backtrack(pos, prev_char, path):
            if pos == W:
                masks.append(tuple(path))
                return
            current_char = row[pos]
            if current_char != '?':
                if current_char == prev_char:
                    return
                path.append(current_char)
                backtrack(pos + 1, current_char, path)
                path.pop()
            else:
                for c in ['1', '2', '3']:
                    if c == prev_char:
                        continue
                    path.append(c)
                    backtrack(pos + 1, c, path)
                    path.pop()
        backtrack(0, None, [])
        valid_masks.append(masks)
        # Early exit if no valid masks exist for this row
        if not masks:
            print(0)
            return

    # Initialize DP with the first row's valid masks
    dp_prev = defaultdict(int)
    for mask in valid_masks[0]:
        dp_prev[mask] = 1

    # Process subsequent rows
    for i in range(1, H):
        dp_current = defaultdict(int)
        current_masks = valid_masks[i]
        # Iterate over all current and previous masks to find compatible transitions
        for mask_current in current_masks:
            total = 0
            for mask_prev, count in dp_prev.items():
                compatible = True
                for j in range(W):
                    if mask_current[j] == mask_prev[j]:
                        compatible = False
                        break
                if compatible:
                    total = (total + count) % MOD
            if total > 0:
                dp_current[mask_current] = total
        dp_prev = dp_current
        # Early exit if no valid configurations exist
        if not dp_prev:
            break

    # Sum all valid configurations
    print(sum(dp_prev.values()) % MOD)

if __name__ == "__main__":
    main()