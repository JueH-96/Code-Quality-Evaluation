def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    x = list(map(int, data[2:]))

    # Step 1: Build toggle lists: toggles[j] will store the query indices (1-based) where x_i == j
    toggles = [[] for _ in range(N+1)]
    for i in range(Q):
        toggles[x[i]].append(i+1)  # store 1-based index of the query

    # Step 2: Determine the size of S after each query (countAfter[i]) by simulating membership
    inS = [False]*(N+1)  # track membership of each element
    countAfter = [0]*(Q+1)  # countAfter[i] = |S| after i-th query
    count = 0
    for i in range(1, Q+1):
        val = x[i-1]
        if not inS[val]:
            inS[val] = True
            count += 1
        else:
            inS[val] = False
            count -= 1
        countAfter[i] = count

    # Step 3: Build prefix sums of countAfter to quickly sum over intervals
    # P[i] = sum of countAfter[1..i], so that sum_{k=a..b} countAfter[k] = P[b] - P[a-1]
    P = [0]*(Q+1)
    for i in range(1, Q+1):
        P[i] = P[i-1] + countAfter[i]

    # Step 4: Compute final A for each j by summing intervals of membership
    A = [0]*(N+1)
    for j in range(1, N+1):
        ts = toggles[j]
        m = len(ts)
        # For each pair (2k+1, 2k+2) in toggles, j is in S from ts[2k]..ts[2k+1]-1
        # If there's a leftover (odd number of toggles), then j is in S from ts[-1]..Q
        idx = 0
        while idx < m:
            start = ts[idx]
            if idx+1 < m:
                end = ts[idx+1] - 1
            else:
                end = Q
            # Add the sum of countAfter from start to end
            if start <= end:
                A[j] += P[end] - P[start-1]
            idx += 2

    # Step 5: Print the result A_1..A_N
    print(" ".join(map(str, A[1:])))
    
# Do not forget to call main()
main()