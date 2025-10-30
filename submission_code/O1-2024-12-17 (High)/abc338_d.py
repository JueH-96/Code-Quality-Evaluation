def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    # Convert islands in X to 0-based indexing
    X = [int(x) - 1 for x in data[2:2+M]]

    # This array will hold a "difference" encoding for how much each edge
    # would add in penalty if that edge is closed. We'll recover actual
    # penalties via prefix sums at the end.
    diff = [0] * (N + 1)

    # A helper function to add weight W to all edges from L to R (inclusive)
    # in clockwise order on our 0-based ring of edges.
    def add_range(L, R, W):
        if L <= R:
            diff[L] += W
            diff[R+1] -= W
        else:
            # The arc wraps past the end of the ring
            diff[L] += W
            diff[N] -= W
            diff[0] += W
            diff[R+1] -= W

    base_dist = 0

    # Process each consecutive pair (X[i], X[i+1]).
    # In a ring of size N, the clockwise distance cwD and
    # the counterclockwise distance is N - cwD. We pick the shorter arc.
    for i in range(M - 1):
        a = X[i]
        b = X[i + 1]
        cwD = (b - a) % N  # clockwise distance
        if cwD <= N - cwD:
            # Clockwise is the shorter arc
            d = cwD
            L = a
            R = (b - 1 + N) % N
        else:
            # Counterclockwise is the shorter arc
            d = N - cwD
            L = b
            R = (a - 1 + N) % N
        
        base_dist += d
        # If an edge on this arc is removed, the distance for this pair
        # would become N - d instead of d. The penalty is (N - d) - d = N - 2d.
        W = N - 2 * d
        # Only add to diff array if there's a positive penalty.
        if W > 0:
            add_range(L, R, W)

    # Now use prefix sums over diff to find, for each edge e, the sum of
    # additional penalties we would incur if edge e were removed.
    curr = 0
    min_penalty = float('inf')
    for i in range(N):
        curr += diff[i]
        if curr < min_penalty:
            min_penalty = curr

    # The final answer is the base distance plus the minimal extra penalty
    print(base_dist + min_penalty)

# Do not forget to call main()!
def _test():
    # You can put any internal tests here if you want,
    # but the problem statement only requires main().
    pass

if __name__ == "__main__":
    main()