def main():
    import sys
    MOD = 998244353
    
    data = sys.stdin.read().strip().split()
    N, M, K = map(int, data[:3])
    xs = []
    ys = []
    idx = 3
    for _ in range(M):
        xi = int(data[idx]) - 1
        yi = int(data[idx + 1]) - 1
        xs.append(xi)
        ys.append(yi)
        idx += 2

    # We will keep all dp-values in a single array "arr" of length N,
    # and use an integer "offset" to interpret which dp-value is stored
    # at which index.  At step t -> t+1 we do:
    #
    #   offset = (offset - 1) mod N
    #
    # which "shifts" dp implicitly, so that dp[t+1][r] = dp[t][r-1]
    # is already in the correct place without having to do O(N) copying.
    #
    # Then for each "shortcut" edge X_j -> Y_j we do:
    #
    #   dp[t+1][Y_j] += dp[t][X_j]
    #
    # translating to array indices as:
    #
    #   arr[offset + Y_j] += arr[(offset + 1) + X_j]
    #
    # (all indices mod N),
    # because offset+1 is the old offset (meaning dp[t]) if offset is dp[t+1].
    
    arr = [0] * N
    arr[0] = 1  # dp[0][0] = 1
    offset = 0
    
    for _ in range(K):
        # Move to next offset (dp[t+1])
        offset = (offset - 1) % N
        base = (offset + 1) % N  # "old" dp[t] offset
        
        # Apply each of the M extra edges
        for j in range(M):
            to_idx = (offset + ys[j]) % N
            from_idx = (base + xs[j]) % N
            val = arr[to_idx] + arr[from_idx]
            if val >= MOD:
                val -= MOD
            arr[to_idx] = val
    
    # Now dp[K] is interpreted at "offset"
    # Sum up dp[K][r] for r = 0..N-1
    ans = 0
    for i in range(N):
        ans += arr[(offset + i) % N]
    ans %= MOD
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()