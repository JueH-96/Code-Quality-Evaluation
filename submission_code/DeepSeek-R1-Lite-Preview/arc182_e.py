import sys
import math

def main():
    import sys
    import math

    N, M, C, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()

    g = math.gcd(C, M)
    period = M // g

    # Compute t_i for each A_i
    t_list = []
    for a in A:
        t_i = (M - a + g - 1) // g  # ceil((M - a)/g)
        t_i = max(t_i, 0)
        t_list.append(t_i)

    # Add a sentinel for the end
    t_list.append(period)

    # Compute the sum
    total_sum = 0
    full_periods = K // period
    remainder = K % period

    # To handle the segments where idx is constant
    prev_t = 0
    for i in range(N):
        t_i = t_list[i]
        if t_i >= period:
            continue
        t_start = t_i
        t_end = t_list[i+1] - 1
        if t_end >= period:
            t_end = period - 1
        if t_start > t_end:
            continue
        # min_t = A[i] + t*g - M
        # Sum over t from t_start to t_end: (A[i] + t*g - M)
        num_t = t_end - t_start + 1
        sum_t = num_t * A[i] + g * (t_start + t_end) * num_t // 2 - M * num_t
        # Multiply by full_periods
        total_sum += sum_t * full_periods
        # Add for the remainder
        if remainder > t_end:
            rem_num = t_end - t_start + 1
            rem_sum = rem_num * A[i] + g * (t_start + t_end) * rem_num // 2 - M * rem_num
            total_sum += rem_sum
        else:
            if t_start <= remainder -1:
                t_upper = min(t_end, remainder -1)
                rem_num = t_upper - t_start + 1
                rem_sum = rem_num * A[i] + g * (t_start + t_upper) * rem_num // 2 - M * rem_num
                total_sum += rem_sum

    print(total_sum)

if __name__ == "__main__":
    main()