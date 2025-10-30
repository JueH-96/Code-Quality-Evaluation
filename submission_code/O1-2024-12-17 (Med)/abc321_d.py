def main():
    import sys
    import bisect
    
    data = sys.stdin.read().strip().split()
    N, M, P = map(int, data[:3])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:]))

    # Precompute sums
    sumA = sum(A)
    sumB = sum(B)

    # Sort B for binary search
    B.sort()

    # Prefix sums of B
    prefixB = [0] * (M + 1)
    for i in range(M):
        prefixB[i+1] = prefixB[i] + B[i]

    # Naive sum of (A_i + B_j) over all pairs
    naive_sum = M * sumA + N * sumB

    # Compute how much we exceed P
    # over_cost = Σ max(0, A_i + B_j - P)
    over_cost = 0

    for x in A:
        # Find how many B_j's are strictly greater than (P - x)
        cutoff = P - x
        idx = bisect.bisect_right(B, cutoff)  # first index where B[idx] > cutoff
        count_greater = M - idx
        if count_greater > 0:
            sum_greater = prefixB[M] - prefixB[idx]   # sum of B_j >= idx..M-1
            over_cost += count_greater * x + sum_greater - count_greater * P

    # The total is naive sum minus the part that exceeds P
    result = naive_sum - over_cost
    print(result)

# Do not forget to call main()
if __name__ == "__main__":
    main()