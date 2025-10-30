import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    # Precompute prefix sums for '1's and '2's
    prefix_1 = [0] * (N + 1)
    prefix_2 = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_1[i] = prefix_1[i - 1] + (1 if S[i - 1] == '1' else 0)
        prefix_2[i] = prefix_2[i - 1] + (1 if S[i - 1] == '2' else 0)

    # Collect positions of '/'
    slashes = []
    for i in range(N):
        if S[i] == '/':
            slashes.append(i + 1)  # store as 1-based index

    output = []
    for _ in range(Q):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1

        # Check if there's any '/' in [L, R]
        left_slash = bisect.bisect_left(slashes, L)
        right_slash = bisect.bisect_right(slashes, R)
        if left_slash >= right_slash:
            output.append("0")
            continue

        A = prefix_1[L - 1]
        B = prefix_2[R]
        total_1 = prefix_1[R] - A
        total_2 = prefix_2[R] - prefix_2[L - 1]
        max_k = min(total_1, total_2, (R - L + 1) // 2)

        low, high, best_k = 0, max_k, 0

        while low <= high:
            mid_k = (low + high) // 2

            # Find lower bound x for current mid_k
            lower = R + 1
            l, h = L, R
            while l <= h:
                m = (l + h) // 2
                if prefix_1[m - 1] >= A + mid_k:
                    lower = m
                    h = m - 1
                else:
                    l = m + 1

            # Find upper bound x for current mid_k
            upper = L - 1
            l, h = L, R
            while l <= h:
                m = (l + h) // 2
                if prefix_2[m] <= B - mid_k:
                    upper = m
                    l = m + 1
                else:
                    h = m - 1

            new_L = max(lower, L)
            new_R = min(upper, R)
            valid = False
            if new_L <= new_R:
                sl_left = bisect.bisect_left(slashes, new_L)
                sl_right = bisect.bisect_right(slashes, new_R)
                if sl_left < sl_right:
                    valid = True

            if valid:
                best_k = mid_k
                low = mid_k + 1
            else:
                high = mid_k - 1

        output.append(str(2 * best_k + 1))

    print('
'.join(output))

if __name__ == "__main__":
    main()