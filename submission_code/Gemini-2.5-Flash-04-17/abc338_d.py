import sys

# Use fast I/O
input = sys.stdin.readline

def solve():
    N, M = map(int, input().split())
    X = list(map(int, input().split()))

    # Calculate the total distance in the original cycle
    # The tour must visit X_1, X_2, ..., X_M in order.
    # The minimum length of the tour in the original cycle is the sum of shortest path distances between consecutive islands X_k, X_{k+1}.
    total_dist_cycle = 0

    # Use a difference array to calculate the total increase in tour length
    # for each bridge removal.
    # Bridges are 1-indexed, from 1 to N. Bridge b connects island b and island (b % N) + 1.
    # Array size N+2 for indices 1 to N+1 to handle range ends.
    diff_inc = [0] * (N + 2) 

    for k in range(M - 1):
        u = X[k]
        v = X[k+1]

        # Calculate increasing and decreasing path distances between u and v in the original cycle
        # d_inc: distance from u to v by increasing index (1 -> 2 -> ... -> N -> 1)
        # 1-based calculation
        if u < v:
            inc_dist = v - u
        else: # u > v
            inc_dist = N - u + v

        # d_dec: distance from u to v by decreasing index (1 -> N -> ... -> 2 -> 1)
        # 1-based calculation
        if u > v:
            dec_dist = u - v
        else: # u < v
            dec_dist = N + u - v

        # Original shortest distance in the cycle
        d_cycle = min(inc_dist, dec_dist)
        total_dist_cycle += d_cycle

        # The increase in distance for the leg (u, v) when the edge on its shortest path is removed
        # is |inc_dist - dec_dist|.
        # If the edge removed is NOT on the shortest path, the distance for this leg remains d_cycle.
        # If the edge removed IS on the shortest path, the new distance is N - d_cycle.
        # The increase is (N - d_cycle) - d_cycle = N - 2 * d_cycle.
        # Note that N - 2 * d_cycle = N - 2 * min(inc_dist, dec_dist)
        # If inc_dist <= dec_dist, increase = N - 2 * inc_dist.
        # If inc_dist > dec_dist, increase = N - 2 * dec_dist.
        # Also, inc_dist + dec_dist = N (if u != v).
        # If inc_dist <= dec_dist, increase = (inc_dist + dec_dist) - 2 * inc_dist = dec_dist - inc_dist = abs(inc_dist - dec_dist).
        # If inc_dist > dec_dist, increase = (inc_dist + dec_dist) - 2 * dec_dist = inc_dist - dec_dist = abs(inc_dist - dec_dist).
        # So the increase is always |inc_dist - dec_dist| if the bridge is on the shortest path.
        delta = abs(inc_dist - dec_dist)

        # Identify the bridges on the shorter path and apply delta using difference array
        # Bridges are 1-indexed, b=1...N. Bridge b connects island b and island (b % N) + 1.
        # Increasing path u -> v uses bridges u, u+1, ..., v-1 (circular).
        # Decreasing path u -> v uses bridges v, v+1, ..., u-1 (circular).

        if inc_dist <= dec_dist: # Shorter path is increasing
            # Bridges on increasing path u -> v: u, u+1, ..., v-1 (circular)
            if u < v: # Range [u, v-1]
                start_bridge = u
                end_bridge = v - 1 # Inclusive
                diff_inc[start_bridge] += delta
                diff_inc[end_bridge + 1] -= delta

            else: # u > v. Range [u, N] U [1, v-1]
                # Range [u, N]
                start_bridge_1 = u
                end_bridge_1 = N
                diff_inc[start_bridge_1] += delta
                diff_inc[end_bridge_1 + 1] -= delta

                # Range [1, v-1]
                start_bridge_2 = 1
                end_bridge_2 = v - 1
                if start_bridge_2 <= end_bridge_2: # Check if range is not empty (v > 1)
                   diff_inc[start_bridge_2] += delta
                   diff_inc[end_bridge_2 + 1] -= delta

        else: # inc_dist > dec_dist. Shorter path is decreasing
            # Bridges on decreasing path u -> v: v, v+1, ..., u-1 (circular)
            if u > v: # Range [v, u-1]
                start_bridge = v
                end_bridge = u - 1
                diff_inc[start_bridge] += delta
                diff_inc[end_bridge + 1] -= delta

            else: # u < v. Range [v, N] U [1, u-1]
                # Range [v, N]
                start_bridge_1 = v
                end_bridge_1 = N
                diff_inc[start_bridge_1] += delta
                diff_inc[end_bridge_1 + 1] -= delta

                # Range [1, u-1]
                start_bridge_2 = 1
                end_bridge_2 = u - 1
                if start_bridge_2 <= end_bridge_2: # Check if range is not empty (u > 1)
                    diff_inc[start_bridge_2] += delta
                    diff_inc[end_bridge_2 + 1] -= delta

    # Compute the actual increase for each bridge by taking prefix sums
    min_increase = float('inf')
    current_increase = 0
    for b in range(1, N + 1):
        current_increase += diff_inc[b]
        min_increase = min(min_increase, current_increase)

    # The minimum possible tour length when one bridge is optimally closed
    # is the original cycle distance plus the minimum total increase over all legs.
    print(total_dist_cycle + min_increase)

solve()