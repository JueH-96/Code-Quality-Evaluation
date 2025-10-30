import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr + N]))
    ptr += N
    m = (N - 1) // 2
    ends = []
    for j in range(m):
        ends.append(A[2 * j + 2])
    # Compute prefix sums
    S = []
    current_sum = 0
    for j in range(m):
        start = A[2 * j + 1]
        end = A[2 * j + 2]
        current_sum += (end - start)
        S.append(current_sum)
    Q = int(data[ptr])
    ptr += 1
    for _ in range(Q):
        l = int(data[ptr])
        r = int(data[ptr + 1])
        ptr += 2
        
        def compute_total_sleep(t):
            j = bisect.bisect_right(ends, t) - 1
            sum_so_far = S[j] if j >= 0 else 0
            if (j + 1) < m:
                next_start = A[2 * (j + 1) + 1]
                next_end = A[2 * (j + 1) + 2]
                if t >= next_start and t <= next_end:
                    sum_so_far += (t - next_start)
            return sum_so_far
        
        total_r = compute_total_sleep(r)
        total_l = compute_total_sleep(l)
        print(total_r - total_l)

if __name__ == '__main__':
    main()