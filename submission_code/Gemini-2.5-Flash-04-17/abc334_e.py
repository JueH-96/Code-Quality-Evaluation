# YOUR CODE HERE
import sys
from collections import deque

# Helper function for modular exponentiation (a^b % mod)
def power(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        base = (base * base) % mod
        exp //= 2
    return res

# Helper function for modular inverse (a^-1 % mod) using Fermat's Little Theorem
def modInverse(n, mod):
    # n must be non-zero and mod must be a prime number
    # For this problem, n (R_count) is guaranteed to be >= 1
    return power(n, mod - 2, mod)

# Check if coordinates are valid (within grid bounds)
def is_valid(r, c, H, W):
    return 0 <= r < H and 0 <= c < W

# Main execution block
if __name__ == "__main__":
    # Read input dimensions
    H, W = map(int, sys.stdin.readline().split())
    # Read grid rows
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Define the modulo
    MOD = 998244353

    # Data structure to store component ID for each green cell
    # Initialize with -1 (unvisited)
    component_id = [[-1] * W for _ in range(H)]
    # Counter for the number of initial green connected components
    C0 = 0
    # Direction vectors for neighbors (up, down, left, right)
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    # 1. Find initial green connected components using BFS
    # Iterate through each cell
    for i in range(H):
        for j in range(W):
            # If the cell is green ('#') and has not been visited yet (component_id is -1)
            if S[i][j] == '#' and component_id[i][j] == -1:
                # Found a new green component
                C0 += 1
                # Start BFS from this cell
                q = deque([(i, j)])
                # Assign the current component ID (C0) to the starting cell
                component_id[i][j] = C0 # Using 1-based indexing for component IDs
                
                # Perform BFS
                while q:
                    r, c = q.popleft()
                    # Check all 4 neighbors
                    for k in range(4):
                        nr, nc = r + dr[k], c + dc[k]
                        # If the neighbor is valid, green, and unvisited
                        if is_valid(nr, nc, H, W) and S[nr][nc] == '#' and component_id[nr][nc] == -1:
                            # Assign the same component ID and add to queue
                            component_id[nr][nc] = C0
                            q.append((nr, nc))

    # 2. Calculate the sum of m values over all red cells
    # m(r,c) is the number of distinct green components adjacent to a red cell (r,c)
    S_sum_m = 0 # Sum of m values
    R_count = 0 # Count of red cells

    # Iterate through each cell again
    for i in range(H):
        for j in range(W):
            # If the cell is red ('.')
            if S[i][j] == '.':
                # Found a red cell, increment red count
                R_count += 1
                # Use a set to store distinct component IDs of green neighbors
                neighbor_components = set()
                # Check all 4 neighbors
                for k in range(4):
                    ni, nj = i + dr[k], j + dc[k]
                    # If the neighbor is valid and green ('#')
                    if is_valid(ni, nj, H, W) and S[ni][nj] == '#':
                        # Add the component ID of the green neighbor to the set
                        neighbor_components.add(component_id[ni][nj])

                # The number of distinct green components adjacent to this red cell is the size of the set
                # Add this count to the total sum S_sum_m, taking modulo
                # len(neighbor_components) is at most 4, S_sum_m can grow large.
                S_sum_m = (S_sum_m + len(neighbor_components)) % MOD

    # 3. Calculate the expected value modulo MOD
    # The expected value is the sum of (components after repainting) over all red cells, divided by the total number of red cells.
    # Components after repainting red cell (r,c) = C0 + 1 - m(r,c)
    # Sum = Sum_{red (r,c)} (C0 + 1 - m(r,c))
    # Sum = Sum_{red (r,c)} (C0 + 1) - Sum_{red (r,c)} m(r,c)
    # Sum = R_count * (C0 + 1) - S_sum_m
    # Expected value = Sum / R_count

    # Numerator P = R_count * (C0 + 1) - S_sum_m
    # Denominator Q = R_count

    # Calculate Numerator P mod MOD
    # R_count and C0+1 are positive integers. Their product fits in Python's integer type.
    # Take modulo after multiplication.
    # The term R_count * (C0 + 1) could be up to 10^6 * (10^6 + 1) ~ 10^12, fits in Python int.
    term_R_C0plus1 = (R_count * (C0 + 1)) % MOD

    # S_sum_m is already calculated modulo MOD
    # Subtract S_sum_m from term_R_C0plus1, ensuring the result is non-negative using MOD
    Numerator_mod = (term_R_C0plus1 - S_sum_m + MOD) % MOD

    # Denominator Q = R_count. Need its modular inverse.
    # R_count is guaranteed to be >= 1, so its inverse exists.
    Denominator_inv = modInverse(R_count, MOD)

    # The final result is Numerator * Denominator_inv mod MOD
    result = (Numerator_mod * Denominator_inv) % MOD

    # Print the result
    print(result)