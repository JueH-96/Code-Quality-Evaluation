def main():
    import sys
    data = sys.stdin.read().strip().split()
    # data[0], data[1] -> N, Q
    # data[2] -> S
    # then pairs (l_i, r_i) follow

    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    
    # Precompute array arr where arr[i] = 1 if S[i] == S[i+1], else 0
    arr = [0] * (N-1)
    for i in range(N - 1):
        if S[i] == S[i+1]:
            arr[i] = 1
    
    # Build prefix sum of arr: prefix[i] = sum of arr up to index i-1 (1-based offset)
    prefix = [0]*(N)
    for i in range(1, N):
        prefix[i] = prefix[i-1] + arr[i-1]
    
    # Process queries
    idx = 3
    out = []
    for _ in range(Q):
        l = int(data[idx]); r = int(data[idx+1])
        idx += 2
        # The number of consecutive duplicates in [l, r] is prefix[r-1] - prefix[l-1]
        # (where prefix is 0-based, but carefully computed to align 1-based indexing)
        out.append(str(prefix[r-1] - prefix[l-1]))
    
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()