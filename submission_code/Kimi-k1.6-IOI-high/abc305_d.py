import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    M = (N - 1) // 2
    intervals = []
    for i in range(M):
        s = A[2 * i + 1]
        e = A[2 * i + 2]
        intervals.append((s, e))
    e_list = [e for s, e in intervals]
    prefix_sum = [0] * (M + 1)
    for i in range(M):
        prefix_sum[i+1] = prefix_sum[i] + (intervals[i][1] - intervals[i][0])
    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr+1])
        ptr += 2
        
        # Calculate total sleep up to r
        k_r = bisect.bisect_right(e_list, r) - 1
        sum_r = prefix_sum[k_r + 1] if k_r >= 0 else 0
        if k_r + 1 < M:
            s, e = intervals[k_r + 1]
            if s <= r < e:
                sum_r += r - s
        
        # Calculate total sleep up to l
        k_l = bisect.bisect_right(e_list, l) - 1
        sum_l = prefix_sum[k_l + 1] if k_l >= 0 else 0
        if k_l + 1 < M:
            s, e = intervals[k_l + 1]
            if s <= l < e:
                sum_l += l - s
        
        print(sum_r - sum_l)

if __name__ == "__main__":
    main()