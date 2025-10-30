def main():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()

    # Read N
    N = int(data[0])

    # Read village coordinates X
    X = list(map(int, data[1:1+N]))

    # Read populations P
    P = list(map(int, data[1+N:1+2*N]))

    # Build prefix sum array for populations
    prefix = [0]*(N+1)
    for i in range(N):
        prefix[i+1] = prefix[i] + P[i]

    # Read number of queries Q
    idx = 1+2*N
    Q = int(data[idx])
    idx += 1

    out = []
    for _ in range(Q):
        # Read L, R
        L, R = int(data[idx]), int(data[idx+1])
        idx += 2

        # Find left boundary using bisect_left for L
        left = bisect.bisect_left(X, L)  # first index where X[i] >= L
        # Find right boundary using bisect_right for R
        right = bisect.bisect_right(X, R)  # first index where X[i] > R

        # Calculate sum of villagers
        total_villagers = prefix[right] - prefix[left]
        out.append(str(total_villagers))

    print('
'.join(out))

# Do not forget to call main!
main()