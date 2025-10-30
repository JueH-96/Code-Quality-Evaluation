def main():
    import sys
    import bisect

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+N]))
    
    sum_A = sum(A)
    R = K - sum_A  # Remaining votes
    
    A_sorted = sorted(A)
    
    # Precompute prefix sums of A_sorted in descending order
    prefix_sum_desc = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sum_desc[i] = prefix_sum_desc[i - 1] + A_sorted[N - i]
    
    result = []
    
    for i in range(N):
        Ai = A[i]
        # Binary search for minimal X
        left = 0
        right = R
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            T = Ai + mid
            # Number of candidates with A_j >= T, excluding A[i]
            pos = bisect.bisect_left(A_sorted, T)
            S = N - pos
            if A_sorted[pos] >= T:
                S = N - pos
            else:
                S = N - (pos + 1)
            if A_sorted[i] >= T:
                S -= 1
            # Calculate sum of first M Y_j_min
            if S >= M:
                current_sum = M
            else:
                m = M - S
                # Sum of (T - A[N - k] + 1) for k=1 to m
                # A[N - k] are the largest m A_j < T
                # Find the m largest A_j < T
                # Since A_sorted is ascending, they are A_sorted[N - m] to A_sorted[N - 1]
                # But need to exclude A[i] if it's among them
                # Find the position of A[i] in A_sorted
                pos_i = bisect.bisect_left(A_sorted, Ai)
                # Now, find the m largest A_j < T, excluding A[i]
                # The candidates with A_j < T are A_sorted[0] to A_sorted[pos-1]
                # We need the m largest among them, excluding A[i] if present
                # Calculate the sum
                if pos - 1 < m:
                    # Not enough candidates with A_j < T
                    current_sum = -1  # Impossible
                else:
                    # Sum of T - A_sorted[pos - m] + 1 to T - A_sorted[pos - 1] + 1
                    # Which is m*T - sum(A_sorted[pos - m] to A_sorted[pos - 1]) + m
                    # Adjust if A[i] is among these m candidates
                    # Find if A[i] is in A_sorted[pos - m to pos - 1]
                    if pos_i < pos and pos_i >= pos - m:
                        # A[i] is among the m candidates, exclude it
                        sum_Y = m * T - (prefix_sum_desc[m] - A_sorted[pos_i]) + m
                    else:
                        sum_Y = m * T - prefix_sum_desc[m] + m
                    current_sum = S + sum_Y
            # Check if current_sum > R - mid
            if current_sum > R - mid:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        result.append(str(ans if ans != -1 else -1))
    print(' '.join(result))

if __name__ == '__main__':
    main()