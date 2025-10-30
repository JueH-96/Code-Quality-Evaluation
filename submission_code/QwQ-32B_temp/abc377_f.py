import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    R = set()
    C = set()
    sum_set = set()
    diff_set = set()

    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        R.add(a)
        C.add(b)
        sum_set.add(a + b)
        diff_set.add(a - b)

    sorted_R = sorted(R)
    sorted_C = sorted(C)
    R_size = len(R)
    C_size = len(C)

    total_initial = (N - R_size) * (N - C_size)

    # Compute A (sum cases)
    A = 0
    for s in sum_set:
        a = max(1, s - N)
        b = min(N, s - 1)
        if a > b:
            continue
        # Count elements in R within [a, b]
        left = bisect.bisect_left(sorted_R, a)
        right = bisect.bisect_right(sorted_R, b)
        count_r = right - left
        T1 = (b - a + 1) - count_r
        # Compute T2
        t2 = 0
        for j in C:
            i_candidate = s - j
            if a <= i_candidate <= b and i_candidate not in R:
                t2 += 1
        count_s = T1 - t2
        A += count_s

    # Compute B (difference cases)
    B = 0
    for d in diff_set:
        a = max(1, d + 1)
        b = min(N, d + N)
        if a > b:
            continue
        # Count elements in R within [a, b]
        left = bisect.bisect_left(sorted_R, a)
        right = bisect.bisect_right(sorted_R, b)
        count_r = right - left
        T1 = (b - a + 1) - count_r
        # Compute T2
        t2 = 0
        for c in C:
            i_candidate = c + d
            if a <= i_candidate <= b and i_candidate not in R:
                t2 += 1
        count_d = T1 - t2
        B += count_d

    # Compute count_AB
    count_AB = 0
    sum_list = list(sum_set)
    diff_list = list(diff_set)
    for s in sum_list:
        for d in diff_list:
            if (s + d) % 2 != 0:
                continue
            i = (s + d) // 2
            j = (s - d) // 2
            if 1 <= i <= N and 1 <= j <= N:
                if i not in R and j not in C:
                    count_AB += 1

    ans = total_initial - (A + B - count_AB)
    print(ans)

if __name__ == "__main__":
    main()