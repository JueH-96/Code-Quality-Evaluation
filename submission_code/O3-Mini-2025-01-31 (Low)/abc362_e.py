def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    mod = 998244353

    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # dp[i] is a dictionary mapping difference d to another dictionary.
    # That inner dictionary maps the subsequence length (>=2) to the number of arithmetic subsequences
    # ending at index i with difference d and of that length.
    dp = [{} for _ in range(N)]
    
    # res[k] will store the total number of arithmetic subsequences of length k.
    # Every single element is an arithmetic sequence so we have N sequences of length 1.
    res = [0] * (N + 1)
    res[1] = N

    # Process each pair i > j to update sequences.
    for i in range(N):
        for j in range(i):
            d = A[i] - A[j]
            
            # Get all sequences ending at index j with difference d.
            prev = dp[j].get(d, {})
            
            # The pair (A[j], A[i]) itself forms a new arithmetic sequence of length 2.
            # We'll add 1 way for that.
            if d not in dp[i]:
                dp[i][d] = {}
            # Increase count for length 2 at i.
            dp[i][d][2] = (dp[i][d].get(2, 0) + 1) % mod
            res[2] = (res[2] + 1) % mod
            
            # For every arithmetic subsequence ending at j with difference d,
            # we can extend it by including A[i].
            for L, count in prev.items():
                new_length = L + 1
                dp[i][d][new_length] = (dp[i][d].get(new_length, 0) + count) % mod
                res[new_length] = (res[new_length] + count) % mod

    # Output res[1]..res[N] as space separated.
    sys.stdout.write(" ".join(str(res[k] % mod) for k in range(1, N+1)))

if __name__ == '__main__':
    main()