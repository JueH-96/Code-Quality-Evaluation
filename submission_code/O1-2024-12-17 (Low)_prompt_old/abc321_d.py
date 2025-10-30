def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M, P = map(int, input_data[:3])
    A = list(map(int, input_data[3:3+N]))
    B = list(map(int, input_data[3+N:]))

    # Sort the arrays
    A.sort()
    B.sort()

    # Precompute prefix sums of B for quick range-sum queries
    prefix_B = [0]*(M+1)
    for i in range(M):
        prefix_B[i+1] = prefix_B[i] + B[i]

    # Sum of all A and sum of all B
    sumA = sum(A)
    sumB = prefix_B[-1]

    # The sum if we took A[i] + B[j] for all pairs (i,j)
    total_sum = sumA*M + sumB*N

    # Now we find how much we subtract because for those pairs where A[i] + B[j] > P:
    # We replace A[i] + B[j] with P. So the difference is (A[i] + B[j]) - P for each such pair.
    # We will compute the total (A[i] + B[j]) for those pairs (sum_exceed) and the count of such pairs (count_exceed).
    # Then the total difference to subtract = sum_exceed - count_exceed * P.
    import bisect
    sum_exceed = 0
    count_exceed = 0

    for x in A:
        threshold = P - x
        # We want all B[j] > threshold
        idx = bisect.bisect_right(B, threshold)
        c = M - idx  # number of B elements that exceed threshold
        if c > 0:
            sum_range = prefix_B[M] - prefix_B[idx]  # sum of those B[idx..]
            sum_exceed += x*c + sum_range
            count_exceed += c

    # Correction term: sum_exceed - (count_exceed)*P
    subtract_amount = sum_exceed - count_exceed * P

    # Final answer
    print(total_sum - subtract_amount)

# Let's call solve() to complete execution
solve()