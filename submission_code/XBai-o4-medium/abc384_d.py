import bisect
from collections import defaultdict
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    S = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + N]))
    ptr += N

    pre_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        pre_sum[i] = pre_sum[i - 1] + A[i - 1]

    T = pre_sum[N]

    # Case 1: Check for subsequences within a single period
    for j in range(1, N + 1):
        target = pre_sum[j] - S
        # Search in pre_sum[0 ... j-1]
        i_pos = bisect.bisect_left(pre_sum, target, 0, j)
        if i_pos < j and pre_sum[i_pos] == target:
            print("Yes")
            return

    # Case 2: Check for subsequences spanning multiple periods
    mod_dict = defaultdict(list)
    for ps in pre_sum:
        rem = ps % T
        mod_dict[rem].append(ps)

    for i in range(1, N + 1):
        suffix = T - pre_sum[i - 1]
        R = S - suffix
        if R < 0:
            continue
        rem_R = R % T
        if rem_R not in mod_dict:
            continue
        lst = mod_dict[rem_R]
        pos = bisect.bisect_right(lst, R)
        if pos > 0:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    main()