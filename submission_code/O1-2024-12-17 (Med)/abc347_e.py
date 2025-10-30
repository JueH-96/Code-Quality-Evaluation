def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    x = list(map(int, input_data[2:]))

    # Step 1: Record for each j which queries toggle j.
    toggles = [[] for _ in range(N+1)]
    for i in range(Q):
        toggles[x[i]].append(i+1)  # using 1-based index for query number

    # Step 2: Compute the size of S after each query in B[1..Q].
    membership = [False] * (N+1)  # track if j is currently in S
    size = 0
    B = [0] * (Q+1)  # B[i] = |S| after the i-th query (1-based)
    
    for i in range(1, Q+1):
        val = x[i-1]  # x is 0-based
        if not membership[val]:
            # insert val
            membership[val] = True
            size += 1
        else:
            # remove val
            membership[val] = False
            size -= 1
        B[i] = size

    # Step 3: Build prefix sums PB so PB[i] = B[1] + ... + B[i].
    PB = [0] * (Q+1)
    for i in range(1, Q+1):
        PB[i] = PB[i-1] + B[i]

    # Step 4: For each j, sum B over the intervals when j is in S.
    A = [0] * N
    for j in range(1, N+1):
        t_list = toggles[j]
        # We toggle membership on queries in t_list.
        # j is in S after the 1st toggle until the 2nd toggle - 1, etc.
        # Because we start "out of S", toggles happen in pairs for "in" intervals.
        # If there's an odd number of toggles, the last interval extends to Q.
        length = len(t_list)
        s = 0
        while s < length:
            start = t_list[s]
            end = (t_list[s+1] - 1) if (s+1 < length) else Q
            # Sum of B in [start .. end] = PB[end] - PB[start-1]
            A[j-1] += PB[end] - PB[start-1]
            s += 2

    # Step 5: Print the result
    print(' '.join(map(str, A)))

# Do not forget to call main!
main()