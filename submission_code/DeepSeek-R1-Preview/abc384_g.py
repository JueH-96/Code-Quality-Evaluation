import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1

    A = list(map(int, input[ptr:ptr + N]))
    ptr += N
    A.sort()

    B = list(map(int, input[ptr:ptr + N]))
    ptr += N
    B.sort()

    # Compute prefix sums for A and B
    prefix_A = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_A[i] = prefix_A[i - 1] + A[i - 1]

    prefix_B = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_B[i] = prefix_B[i - 1] + B[i - 1]

    K = int(input[ptr])
    ptr += 1

    for _ in range(K):
        X = int(input[ptr])
        Y = int(input[ptr + 1])
        ptr += 2

        # Get the first X elements of sorted A and Y elements of sorted B
        S = A[:X]
        T = B[:Y]

        sum_S = prefix_A[X]
        sum_T = prefix_B[Y]

        # Precompute the prefix sums for T
        prefix_T = [0] * (Y + 1)
        for i in range(1, Y + 1):
            prefix_T[i] = prefix_T[i - 1] + T[i - 1]

        term1 = 0
        term2 = 0

        for s in S:
            # Find the number of elements in T <= s
            pos = bisect.bisect_right(T, s)
            cnt_le = pos
            sum_le = prefix_T[pos]
            term1 += s * cnt_le
            term2 += sum_le

        sum_total = 2 * term1 - Y * sum_S + X * sum_T - 2 * term2
        print(sum_total)

if __name__ == '__main__':
    main()