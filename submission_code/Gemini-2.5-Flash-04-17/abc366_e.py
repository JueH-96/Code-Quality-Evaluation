import sys
from collections import Counter
import bisect

# Function to compute F or G array
# coords: list of x_i or y_i coordinates
# D: maximum allowed sum
# N: number of points
def compute_fg_array(coords, D, N):
    # Constraints state 1 <= N, so coords list is not empty.
    # Handle the edge case just in case, although not strictly necessary per constraints.
    if not coords:
        return [0] * (D + 1)

    # Sort coordinates
    sorted_coords = sorted(coords)

    # Get unique sorted coordinates and their counts
    # Using Counter and sorting keys is a convenient way
    counts = Counter(sorted_coords)
    unique_coords = sorted(counts.keys())
    unique_counts = [counts[u] for u in unique_coords]

    # Determine calculation range for the coordinate
    # The range [min_coord - D - 1, max_coord + D + 1] is guaranteed
    # to contain all integer points whose sum of distances f(coord) is <= D.
    # Proof sketch: f(x) = sum |x - x_i| >= |sum (x - x_i)| = |Nx - sum(x_i)|.
    # If f(x) <= D, then |Nx - sum(x_i)| <= D.
    # -D <= Nx - sum(x_i) <= D
    # sum(x_i) - D <= Nx <= sum(x_i) + D
    # (sum(x_i) - D)/N <= x <= (sum(x_i) + D)/N.
    # The range [min_coord - D - 1, max_coord + D + 1] covers this interval.
    # If x < min_coord - D - 1, then for any x_i, x < min_coord <= x_i.
    # |x - x_i| = x_i - x > x_i - (min_coord - D - 1) = (x_i - min_coord) + D + 1 >= D + 1.
    # f(x) = sum |x - x_i| > N * (D + 1). If N >= 1 and D >= 0, this is > D.
    # Similarly for x > max_coord + D + 1.
    min_coord = sorted_coords[0]
    max_coord = sorted_coords[-1]
    start_coord = min_coord - D - 1
    end_coord = max_coord + D + 1

    # Array to store F[k] or G[k], size D+1.
    # fg_array[k] will store the number of integer coordinates (x or y)
    # for which the sum of distances to the input points is exactly k.
    # Using Python's built-in int which supports arbitrary precision.
    fg_array = [0] * (D + 1)

    # Initialize current coordinate for iteration
    current_coord = start_coord

    # Compute initial sum of distances for the start coordinate
    # This sum can be large, use Python's default int.
    current_sum_dist = sum(abs(current_coord - c) for c in sorted_coords)

    # count_le_prev_coord: Number of points in `coords` less than or equal to `current_coord - 1`.
    # This is C(current_coord - 1). It is used to calculate the slope.
    # f(x+1) - f(x) = 2 * C(x) - N, where C(x) is count of coords <= x.
    # We maintain C(x-1) as `count_le_prev_coord` in the loop for coordinate x.
    count_le_prev_coord = bisect.bisect_right(sorted_coords, current_coord - 1)

    # u_idx: Index into `unique_coords`.
    # It points to the index of the first unique coordinate that is >= `current_coord`.
    u_idx = bisect.bisect_left(unique_coords, current_coord)

    # Iterate through the relevant range of coordinates
    while current_coord <= end_coord:
        # If the current sum of distances is within the limit D
        # sum of absolute values is always non-negative.
        if current_sum_dist <= D:
            fg_array[current_sum_dist] += 1

        # --- Prepare for the next iteration (current_coord + 1) ---

        # Count of points in `coords` exactly equal to `current_coord`.
        # Check if the unique coordinate at u_idx is equal to the current_coord
        count_eq_current_coord = 0
        if u_idx < len(unique_coords) and unique_coords[u_idx] == current_coord:
            # If it is, the count is unique_counts[u_idx]
            count_eq_current_coord = unique_counts[u_idx]

        # Count of points in `coords` less than or equal to `current_coord`.
        # C(x) = C(x-1) + (count == x)
        count_le_current_coord = count_le_prev_coord + count_eq_current_coord

        # Slope f(x+1) - f(x) = 2 * C(x) - N = 2 * count_le_current_coord - N
        # This slope tells us how the sum of distances changes when moving from x to x+1.
        slope = 2 * count_le_current_coord - N

        # Update sum of distances for the next coordinate (current_coord + 1)
        current_sum_dist += slope

        # Update count_le_prev_coord for the next iteration (at current_coord + 1):
        # The count of points <= (current_coord + 1) - 1 is the count of points <= current_coord.
        # So, `count_le_prev_coord` for the next iteration becomes `count_le_current_coord`.
        count_le_prev_coord = count_le_current_coord

        # Move to the next coordinate
        current_coord += 1

        # Update u_idx: If we just processed the unique coordinate at the previous `current_coord`
        # (which means `count_eq_current_coord > 0`), then for the new `current_coord`, `u_idx`
        # should advance to point to the next unique coordinate.
        # The check `unique_coords[u_idx] == current_coord` (checked before incrementing `current_coord`)
        # was true if `count_eq_current_coord > 0`. So, after incrementing `current_coord`,
        # `u_idx` needs to move past the unique coordinate that was equal to the old `current_coord`.
        if count_eq_current_coord > 0:
             u_idx += 1

        # The loop condition `current_coord <= end_coord` checks if the current coordinate
        # we just finished processing was within the range. The loop continues as long
        # as the next coordinate to process (current_coord + 1) is potentially within or beyond end_coord.
        # The `while current_coord <= end_coord:` means we process `start_coord` up to `end_coord`.
        # The last iteration will process `current_coord == end_coord`. After this, `current_coord` becomes `end_coord + 1`.
        # The next loop check `current_coord <= end_coord` will be false, and the loop terminates.

    return fg_array


def solve():
    # Read input N and D
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    D = int(line1[1])

    # Read the points (x_i, y_i)
    x_coords = []
    y_coords = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        x_coords.append(x)
        y_coords.append(y)

    # Compute F array: F[k] = number of integer x such that sum_i |x - x_i| = k
    # f(x) = sum_i |x - x_i|. We count integer x values for each possible value of f(x) up to D.
    F = compute_fg_array(x_coords, D, N)

    # Compute G array: G[k] = number of integer y such that sum_i |y - y_i| = k
    # g(y) = sum_i |y - y_i|. We count integer y values for each possible value of g(y) up to D.
    G = compute_fg_array(y_coords, D, N)

    # We need to find the number of integer pairs (x, y) such that
    # sum_i |x - x_i| + sum_i |y - y_i| <= D.
    # Let f(x) = sum_i |x - x_i| and g(y) = sum_i |y - y_i|.
    # We need the number of pairs (x, y) such that f(x) + g(y) <= D.
    # This is sum over all possible integer values kx, ky such that kx + ky <= D,
    # of (number of integer x with f(x)=kx) * (number of integer y with g(y)=ky).
    # This sum is: sum_{kx=0..D} sum_{ky=0..D-kx} F[kx] * G[ky].

    # To compute the inner sum efficiently, we use prefix sums of G.
    # Let S_G[k] = sum_{j=0..k} G[j].
    # S_G[k] is the number of integer y such that g(y) <= k.
    S_G = [0] * (D + 1)
    # Constraint guarantees 0 <= D. If D=0, the loop range(1, D+1) is empty.
    if D >= 0:
        S_G[0] = G[0]
        for k in range(1, D + 1):
            S_G[k] = S_G[k - 1] + G[k]

    # The total answer is sum_{kx=0..D} F[kx] * (sum_{ky=0..D-kx} G[ky])
    # The inner sum is sum_{ky=0..D-kx} G[ky] = S_G[D - kx].
    # The formula becomes: ans = sum_{kx=0..D} F[kx] * S_G[D - kx].
    ans = 0
    for kx in range(D + 1):
        # We need F[kx] * S_G[D - kx].
        # Since 0 <= kx <= D, the index D - kx is always between 0 and D,
        # which are valid indices for S_G (size D+1).
        # Also, kx is always a valid index for F (size D+1).
        ans += F[kx] * S_G[D - kx]

    # Print the final answer
    print(ans)

# Call the solve function to run the program
solve()