def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    maxA = max(A)
    
    # Frequency array
    freq = [0]*(maxA+1)
    for val in A:
        freq[val] += 1
    
    # Build prefix sum of freq for fast range-sum queries
    # P[i] = freq[1] + freq[2] + ... + freq[i]
    # We'll use 1-based indexing for convenience
    P = [0]*(maxA+1)
    for i in range(1, maxA+1):
        P[i] = P[i-1] + freq[i]
    
    # 1) Contribution from pairs of the same value:
    #    For each v, number of pairs is freq[v]*(freq[v]-1)//2,
    #    each pair contributes floor(v/v) = 1
    #    so total_same = sum( freq[v]*(freq[v]-1)//2 ).
    total_same = 0
    for v in range(1, maxA+1):
        c = freq[v]
        if c > 1:
            total_same += c*(c-1)//2
    
    # 2) Contribution from pairs of different values (x<y):
    #    We sum freq[x]*freq[y]*floor(y/x) for y>x.
    #    Instead of i<j over all, do an outer loop over x, then
    #    for each integer n = floor(y/x), consider y in [n*x, (n+1)*x - 1]
    #    but only y > x. So L = max(x+1, n*x), R = (n+1)*x - 1.
    #    We'll sum freq[y] over that range quickly using P[] array.
    
    total_diff = 0
    
    # Collect all distinct x in ascending order to skip freq=0
    # (This helps skip unneeded loops.)
    distinct_vals = []
    for x in range(1, maxA+1):
        if freq[x] > 0:
            distinct_vals.append(x)
    
    # Main loop
    for x in distinct_vals:
        f = freq[x]
        if f == 0:
            continue
        # We'll iterate n so that y = n*x to (n+1)*x - 1 does not exceed maxA
        # i.e. n*x <= maxA => n <= maxA//x
        # We'll break whenever the range is empty or out of bounds
        limit = maxA // x
        # n goes from 1..limit
        # For each n, we define L, R
        # L = max(x+1, n*x), R = min(maxA, (n+1)*x - 1)
        # If L>R, skip
        for n in range(1, limit+1):
            L = n*x
            if L <= x:  # we need y > x
                L = x+1
            R = (n+1)*x - 1
            if R > maxA:
                R = maxA
            if L > R:
                continue
            # sum of freq[y] for y in [L..R]
            s = P[R] - P[L-1]
            total_diff += f * n * s
    
    # Total is sum of same-value pairs plus different-value pairs
    answer = total_same + total_diff
    
    print(answer)

# Call main() as required
if __name__ == "__main__":
    main()