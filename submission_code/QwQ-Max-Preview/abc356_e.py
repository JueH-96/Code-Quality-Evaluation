import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    A.sort()
    if not A:
        print(0)
        return
    max_A = A[-1]
    max_t = max_A + 1
    freq = [0] * (max_t + 2)
    for a in A:
        freq[a] += 1
    # Compute suffix sums
    suffix_sum = [0] * (max_t + 2)
    for t in range(max_t, 0, -1):
        suffix_sum[t] = suffix_sum[t + 1] + freq[t]
    total = 0
    for i in range(N):
        x = A[i]
        if x == 0:
            continue
        max_q = max_A // x
        sum_x = 0
        for k in range(1, max_q + 1):
            t = k * x
            if t > max_t:
                count_in_suffix = 0
            else:
                count_in_suffix = suffix_sum[t]
            if t > x:
                count_in_prefix = 0
            else:
                pos = bisect.bisect_left(A, t, 0, i + 1)
                count_in_prefix = (i + 1) - pos
            sum_x += (count_in_suffix - count_in_prefix)
        total += sum_x
    print(total)

if __name__ == '__main__':
    main()