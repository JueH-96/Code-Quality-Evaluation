def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Positions of each value
    # Since A_i ≤ N, we can create a list of lists of size N+1
    positions = [[] for _ in range(N+1)]
    for i, val in enumerate(A):
        positions[val].append(i)

    # 1) Compute F = number of (i,j,k) with i<j<k and A_i = A_k
    #    For each value v, let p1 < p2 < ... < pm be the positions of v.
    #    Then for each pair (p[a], p[b]) with a < b, we can choose j in (p[a], p[b]) in (p[b] - p[a] - 1) ways.
    #    Summation: sum_{b=1..m-1} sum_{a=0..b-1} (p[b] - p[a] - 1).
    #    We can do this in O(m) time per value with prefix sums.
    totalF = 0
    for val_positions in positions:
        m = len(val_positions)
        if m < 2:
            continue
        # Build prefix sums for these positions
        prefix_sum = [0] * (m+1)
        for i in range(m):
            prefix_sum[i+1] = prefix_sum[i] + val_positions[i]
        # Calculate contribution
        # sum_{b=1..m-1} [ b*p[b] - sum_{a=0..b-1}(p[a]) - b ]
        # = sum_{b=1..m-1} ( b * p[b] - prefix_sum[b] - b )
        contrib = 0
        for b in range(1, m):
            pb = val_positions[b]
            contrib += b * pb - prefix_sum[b] - b
        totalF += contrib

    # 2) Compute G = number of (i,j,k) with i<j<k and A_i = A_j = A_k
    #    For each value v with frequency m, the count is C(m, 3) = m*(m-1)*(m-2)//6.
    def comb3(x):
        if x < 3:
            return 0
        return x*(x-1)*(x-2)//6

    totalG = 0
    for val_positions in positions:
        m = len(val_positions)
        totalG += comb3(m)

    # The desired answer = F - G
    print(totalF - totalG)

# Don't forget to call main!
if __name__ == "__main__":
    main()