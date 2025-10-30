import bisect
import sys
import math

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    C = int(input[ptr]); ptr += 1
    K = int(input[ptr]); ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    sorted_A = sorted(A)
    g = math.gcd(C, M)
    T = M // g

    # Generate critical points
    critical = [0]
    for a in sorted_A:
        if a > critical[-1]:
            critical.append(a)
    critical.append(M)

    sum_cycle = 0

    for j in range(len(critical) - 1):
        L = critical[j]
        R = critical[j + 1]

        A_val = M - R
        B_val = M - L

        low = max(A_val, 0)
        high = min(B_val, M)
        count_s = 0
        if low < high:
            count_s = ( (high - 1) // g ) - ( (low - 1) // g )
        case1 = 1 if (L <= 0 < R) else 0
        total_count = count_s + case1

        # compute sum_s for case2
        sum_s_case2 = 0
        if low < high:
            k_min = (low + g - 1) // g
            k_max = (high - 1) // g
            if k_min <= k_max:
                first = k_min * g
                last = k_max * g
                n = k_max - k_min + 1
                sum_s_case2 = (first + last) * n // 2
            else:
                sum_s_case2 = 0
        else:
            sum_s_case2 = 0

        sum_t_case2 = count_s * M - sum_s_case2
        sum_t_case1 = 0  # since t=0 contributes 0
        sum_t = sum_t_case2 + (0 if not case1 else 0)  # case1's t is 0

        if R == M:
            contribution = total_count * (M + sorted_A[0]) - sum_t
        else:
            contribution = total_count * R - sum_t
        sum_cycle += contribution

    total_cycles = K // T
    remaining = K % T

    sum_total = total_cycles * sum_cycle

    # Compute sum_remaining
    sum_remaining = 0
    for k in range(remaining):
        x = (C * k) % M
        threshold = M - x
        pos = bisect.bisect_left(sorted_A, threshold)
        case_a = float('inf')
        if pos > 0:
            case_a = x + sorted_A[0]
        case_b = float('inf')
        if pos < len(sorted_A):
            case_b = sorted_A[pos] - threshold
        min_val = min(case_a, case_b)
        sum_remaining += min_val

    sum_total += sum_remaining
    print(sum_total)

if __name__ == '__main__':
    main()