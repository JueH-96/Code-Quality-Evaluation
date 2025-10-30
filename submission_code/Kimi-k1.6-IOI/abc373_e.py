import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    sum_A = sum(A)
    R = K - sum_A

    # Sort the list of (A[i], i)
    sorted_list = sorted((A[i], i) for i in range(N))
    sorted_A_vals = [x[0] for x in sorted_list]
    pos = [0] * N
    for i in range(N):
        pos[sorted_list[i][1]] = i

    # Compute prefix sum
    pre_sum = [0] * (N + 1)
    for i in range(N):
        pre_sum[i + 1] = pre_sum[i] + sorted_A_vals[i]

    result = []
    for i in range(N):
        ai = A[i]
        low = 0
        high = R
        ans = -1
        while low <= high:
            mid = (low + high) // 2
            X = mid
            S = ai + X
            rem = R - X
            if rem < 0:
                low = mid + 1
                continue

            # Compute count_over
            idx = bisect.bisect_right(sorted_A_vals, S)
            p = pos[i]
            original_count_over = N - idx
            if sorted_A_vals[p] > S:
                count_over = original_count_over - 1
            else:
                count_over = original_count_over

            # Compute m
            original_le = idx
            if p < idx:
                m = original_le - 1
            else:
                m = original_le

            if m == 0:
                possible_k = 0
                sum_req = 0
            else:
                left = 0
                right = m
                best_k = 0
                while left <= right:
                    k = (left + right) // 2
                    a = idx - k
                    b = idx - 1
                    if a < 0:
                        a = 0
                    # Compute sum_original
                    sum_original = pre_sum[b + 1] - pre_sum[a]
                    # Check if p is in [a, b]
                    in_range = False
                    if p >= a and p <= b:
                        in_range = True
                    if in_range:
                        if a > 0:
                            sum_subset = sum_original - sorted_A_vals[p] + sorted_A_vals[a - 1]
                        else:
                            sum_subset = sum_original - sorted_A_vals[p]
                    else:
                        sum_subset = sum_original
                    sum_req_k = k * (S + 1) - sum_subset
                    if sum_req_k <= rem:
                        best_k = k
                        left = k + 1
                    else:
                        right = k - 1
                possible_k = best_k

            total = count_over + possible_k
            if total < M:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1

        result.append(ans if (ans != -1 and ans <= R) else -1)

    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()