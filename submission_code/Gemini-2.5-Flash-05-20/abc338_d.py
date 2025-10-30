# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))

    # Convert islands to 0-indexed
    X_0indexed = [x - 1 for x in X]

    # Initialize difference array for penalties
    # penalty_for_bridge[i] will store the sum of penalties if bridge i (connecting island i and (i+1)%N) is removed
    penalty_for_bridge_diff = [0] * N

    total_initial_shortest_length = 0

    for k in range(M - 1):
        u = X_0indexed[k]
        v = X_0indexed[k+1]

        # Calculate clockwise and counter-clockwise distances
        d_cw = (v - u + N) % N
        d_ccw = (u - v + N) % N

        shortest_len = min(d_cw, d_ccw)
        total_initial_shortest_length += shortest_len

        # Calculate the penalty if this segment's shortest path is broken
        segment_penalty = N - 2 * shortest_len

        # Determine the range of bridges on the shortest path
        # Bridges are indexed 0 to N-1, where bridge i connects island i and (i+1)%N

        if d_cw <= d_ccw: # Clockwise path is shortest: u -> u+1 -> ... -> v
            # Bridges involved: u, (u+1)%N, ..., (v-1+N)%N
            # Start bridge index: u
            # End bridge index: (v-1+N)%N
            L = u
            R = (v - 1 + N) % N
        else: # Counter-clockwise path is shortest: u -> u-1 -> ... -> v
            # Bridges involved: (u-1+N)%N, (u-2+N)%N, ..., v
            # Start bridge index: v
            # End bridge index: (u-1+N)%N
            L = v
            R = (u - 1 + N) % N

        # Apply segment_penalty to the range [L, R] in penalty_for_bridge_diff
        if L <= R:
            # Normal range: [L, R]
            penalty_for_bridge_diff[L] += segment_penalty
            if R + 1 < N:
                penalty_for_bridge_diff[R + 1] -= segment_penalty
        else:
            # Wrap-around range: [L, N-1] and [0, R]
            penalty_for_bridge_diff[L] += segment_penalty
            penalty_for_bridge_diff[0] += segment_penalty # This applies value to [0, R]
            if R + 1 < N: # This ends the effect for [0, R] segment
                penalty_for_bridge_diff[R + 1] -= segment_penalty
            # The effect for [L, N-1] naturally terminates at N-1 for prefix sum.

    # Calculate actual penalties for each bridge using prefix sums
    min_total_penalty = float('inf')
    current_total_penalty = 0

    for i in range(N):
        current_total_penalty += penalty_for_bridge_diff[i]
        min_total_penalty = min(min_total_penalty, current_total_penalty)

    print(total_initial_shortest_length + min_total_penalty)

solve()