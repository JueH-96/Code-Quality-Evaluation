import sys

# Modular arithmetic
MOD = 998244353

def add(a, b):
    return (a + b) % MOD

def sub(a, b):
    return (a - b + MOD) % MOD

def mul(a, b):
    return (a * b) % MOD

# Function to compute DP from (1,1) up to (target_h, target_w)
# Returns dp[target_h][target_w]
def calculate_dp_val(grid, H, W, target_h, target_w):
    # This function computes the sum of products along paths from (1,1) to (target_h, target_w)
    # including the value at (target_h, target_w).
    # It uses a temporary DP table for the subgrid (1,1) to (target_h, target_w).

    if target_h < 1 or target_w < 1:
        return 0 # Should not be called with target < 1 in correct logic

    # temp_dp table size (target_h + 1) x (target_w + 1) for 1-based indexing
    temp_dp = [[0] * (target_w + 1) for _ in range(target_h + 1)]

    # Base case
    temp_dp[1][1] = grid[1][1]

    # First row (up to target_w)
    for w in range(2, target_w + 1):
        temp_dp[1][w] = mul(temp_dp[1][w-1], grid[1][w])

    # First column (up to target_h)
    for h in range(2, target_h + 1):
        temp_dp[h][1] = mul(temp_dp[h-1][1], grid[h][1])

    # Rest of the subgrid
    for h in range(2, target_h + 1):
        for w in range(2, target_w + 1):
            temp_dp[h][w] = mul(add(temp_dp[h-1][w], temp_dp[h][w-1]), grid[h][w])

    return temp_dp[target_h][target_w]

# Function to compute DP from (start_h, start_w) down to (H,W)
# Returns dp'[start_h][start_w]
def calculate_dp_prime_val(grid, H, W, start_h, start_w):
    # This function computes the sum of products along paths from (start_h, start_w) to (H,W)
    # including the value at (start_h, start_w).
    # It uses a temporary DP table for the subgrid (start_h, start_w) to (H,W).

    if start_h > H or start_w > W:
        return 0 # Should not be called with start > H or W in correct logic

    # temp_dp_prime table size (H - start_h + 2) x (W - start_w + 2) for 1-based indexing convenience relative to subgrid
    # Original grid (h, w) corresponds to temp_dp_prime (h - start_h + 1, w - start_w + 1)
    temp_H_dim = H - start_h + 1
    temp_W_dim = W - start_w + 1
    temp_dp_prime = [[0] * (temp_W_dim + 1) for _ in range(temp_H_dim + 1)]

    # Base case: dp'[H][W] -> temp_dp_prime[H - start_h + 1][W - start_w + 1]
    temp_dp_prime[temp_H_dim][temp_W_dim] = grid[H][W]

    # Last row (from W down to start_w) in original grid terms
    for w_orig in range(W - 1, start_w - 1, -1):
        w_rel = w_orig - start_w + 1
        temp_dp_prime[temp_H_dim][w_rel] = mul(temp_dp_prime[temp_H_dim][w_rel + 1], grid[H][w_orig])

    # Last column (from H down to start_h) in original grid terms
    for h_orig in range(H - 1, start_h - 1, -1):
        h_rel = h_orig - start_h + 1
        temp_dp_prime[h_rel][temp_W_dim] = mul(temp_dp_prime[h_rel + 1][temp_W_dim], grid[h_orig][W])

    # Rest of the subgrid (from H-1, W-1 down to start_h, start_w) in original grid terms
    for h_orig in range(H - 1, start_h - 1, -1):
        for w_orig in range(W - 1, start_w - 1, -1):
            h_rel = h_orig - start_h + 1
            w_rel = w_orig - start_w + 1
            temp_dp_prime[h_rel][w_rel] = mul(add(temp_dp_prime[h_rel + 1][w_rel], temp_dp_prime[h_rel][w_rel + 1]), grid[h_orig][w_orig])

    # The value for dp'[start_h][start_w] is at temp_dp_prime[1][1]
    return temp_dp_prime[1][1]


def main():
    H, W = map(int, sys.stdin.readline().split())
    # Use 1-based indexing for the grid A
    A = [[0] * (W + 1)]
    for _ in range(H):
        A.append([0] + list(map(int, sys.stdin.readline().split())))

    Q, current_sh, current_sw = map(int, sys.stdin.readline().split())

    # Calculate initial total sum S_initial = dp[H][W]
    # Use a full DP table for initial calculation
    dp_initial = [[0] * (W + 1) for _ in range(H + 1)]
    dp_initial[1][1] = A[1][1]
    for w in range(2, W + 1):
        dp_initial[1][w] = mul(dp_initial[1][w-1], A[1][w])
    for h in range(2, H + 1):
        dp_initial[h][1] = mul(dp_initial[h-1][1], A[h][1])
    for h in range(2, H + 1):
        for w in range(2, W + 1):
            dp_initial[h][w] = mul(add(dp_initial[h-1][w], dp_initial[h][w-1]), A[h][w])

    current_total_sum = dp_initial[H][W]

    for _ in range(Q):
        d, a = sys.stdin.readline().split()
        a = int(a)

        # Position before the move
        # old_sh, old_sw = current_sh, current_sw # Not strictly needed

        # Update current position based on the move
        if d == 'U':
            current_sh -= 1
        elif d == 'D':
            current_sh += 1
        elif d == 'L':
            current_sw -= 1
        elif d == 'R':
            current_sw += 1

        # New position after the move is the cell being updated
        new_sh, new_sw = current_sh, current_sw

        old_A_value = A[new_sh][new_sw]

        # Calculate C(new_sh, new_sw) based on the grid *before* this update
        # C(sh,sw) = DP_1,1->sh,sw_excl * DP_sh,sw->H,W_excl
        # DP_1,1->sh,sw_excl is sum of products along paths from (1,1) to (sh,sw), *excluding* A[sh][sw]
        # It is the sum of products along paths ending at (sh-1,sw) or (sh,sw-1), including values there.
        # L = dp[sh-1][sw] + dp[sh][sw-1] (using dp def that includes last cell)

        L = 0
        if new_sh == 1 and new_sw == 1:
             L = 1 # Path (1,1) to (1,1) excluding (1,1) has product 1 (empty product)
        else:
             # Paths from (1,1) to (new_sh, new_sw) must arrive from (new_sh-1, new_sw) or (new_sh, new_sw-1)
             # Sum of products along paths (1,1) to (new_sh, new_sw) excluding A[new_sh][new_sw]
             # = (Sum product paths (1,1) to (new_sh-1, new_sw)) + (Sum product paths (1,1) to (new_sh, new_sw-1))
             # Note: calculate_dp_val returns sum product including the target cell.
             # So, need dp[new_sh-1][new_sw] and dp[new_sh][new_sw-1] from previous calculation.
             if new_sh > 1:
                 L = add(L, calculate_dp_val(A, H, W, new_sh - 1, new_sw))
             if new_sw > 1:
                 L = add(L, calculate_dp_val(A, H, W, new_sh, new_sw - 1))

        # DP_sh,sw->H,W_excl is sum of products along paths from (sh,sw) to (H,W), *excluding* A[sh][sw]
        # It is the sum of products along paths starting from (sh+1,sw) or (sh,sw+1), including values there.
        # R = dp'[sh+1][sw] + dp'[sh][sw+1] (using dp' def that includes first cell)

        R = 0
        if new_sh == H and new_sw == W:
            R = 1 # Path (H,W) to (H,W) excluding (H,W) has product 1
        else:
            # Paths from (new_sh, new_sw) to (H,W) must start by moving to (new_sh+1, new_sw) or (new_sh, new_sw+1)
            # Sum of products along paths (new_sh, new_sw) to (H,W) excluding A[new_sh][new_sw]
            # = (Sum product paths (new_sh+1, new_sw) to (H,W)) + (Sum product paths (new_sh, new_sw+1) to (H,W))
            # Note: calculate_dp_prime_val returns sum product including the start cell.
            # So, need dp'[new_sh+1][new_sw] and dp'[new_sh][new_sw+1] from subsequent calculation.
            if new_sh < H:
                R = add(R, calculate_dp_prime_val(A, H, W, new_sh + 1, new_sw))
            if new_sw < W:
                R = add(R, calculate_dp_prime_val(A, H, W, new_sh, new_sw + 1))

        C = mul(L, R)

        # Update the total sum
        # S_new = S_old + C_old * (A_new - A_old)
        # C_old is calculated based on A values *before* update
        diff = sub(a, old_A_value)
        current_total_sum = add(current_total_sum, mul(C, diff))

        # Update the grid value A[new_sh][new_sw] = a
        A[new_sh][new_sw] = a

        # Print the new total sum
        print(current_total_sum)


if __name__ == "__main__":
    main()