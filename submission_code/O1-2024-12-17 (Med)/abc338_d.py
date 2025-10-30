def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    X = list(map(int, input_data[2:]))

    # A helper to compute the edge index for the island i (1-based).
    # Edge i connects island i and i+1 (in clockwise order), 
    # and edge N connects island N and island 1.
    def edge_index(i):
        # If i < N, the edge is i.
        # If i == N, the edge is N (connecting island N -> 1).
        return i if i < N else N

    # We'll maintain a difference array for the "extra cost" contributions.
    # diff[e] will track how much E_i is added to edge e.
    # We'll store this in 1-based indexing up to N+1 for ease of prefix computations.
    diff = [0]*(N+2)

    total_d = 0  # Sum of ring-distances of all consecutive X_i (when no bridge is closed)

    # Process consecutive pairs
    for i in range(M-1):
        a, b = X[i], X[i+1]
        # Compute clockwise distance from a to b
        if b >= a:
            c = b - a
        else:
            c = b - a + N
        # c in [1..N-1] because X[i] != X[i+1]

        # If 2*c == N, the distance is N/2 and there is no unique path
        # => removing one edge won't force us onto a longer path, so E_i=0.
        if 2*c == N:
            # distance = N/2
            d = N//2
            total_d += d
            # No update to diff because E_i=0 for this pair
            continue

        # Otherwise, there's a unique shortest path
        # If 2*c < N => we use c in clockwise direction
        # If 2*c > N => we use (N-c) in the other direction
        if 2*c < N:
            # Shortest path length = c, E_i = N - 2c
            d = c
            E = N - 2*c
            total_d += d
            # The path is from a to b (clockwise)
            start_edge = edge_index(a)
            length = d  # number of edges on this path
        else:
            # 2*c > N
            # Shortest path length = N-c, E_i = 2c - N
            d = N - c
            E = 2*c - N
            total_d += d
            # The path is from b to a (clockwise)
            start_edge = edge_index(b)
            length = d

        if E > 0:
            # Mark in the difference array the segment of length "length" starting at "start_edge"
            end_edge = start_edge + length - 1
            if end_edge <= N:
                diff[start_edge] += E
                diff[end_edge+1] -= E
            else:
                # wrap around
                diff[start_edge] += E
                diff[N+1] -= E
                wrap_end = end_edge - N
                diff[1] += E
                diff[wrap_end+1] -= E

    # Now compute the prefix sums to get how much "extra cost" each edge would incur if removed
    curr = 0
    min_val = 10**18  # large sentinel
    for i in range(1, N+1):
        curr += diff[i]
        if curr < min_val:
            min_val = curr

    # The final answer is the base sum of distances plus the minimal additional cost
    # that occurs by removing the best-chosen edge.
    answer = total_d + min_val
    print(answer)

# Call main() at the end
if __name__ == '__main__':
    main()