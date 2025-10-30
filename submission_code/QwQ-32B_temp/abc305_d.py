import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    m = (N - 1) // 2
    intervals = []
    ends = []
    for k in range(m):
        start = A[2*k + 1]
        end = A[2*k + 2]
        intervals.append((start, end))
        ends.append(end)
    
    # Precompute prefix sums
    pre_sum = [0] * (m + 1)
    for i in range(m):
        pre_sum[i+1] = pre_sum[i] + (intervals[i][1] - intervals[i][0])
    
    Q = int(input[ptr])
    ptr += 1
    
    def compute(x):
        if m == 0:
            return 0
        if x < intervals[0][0]:
            return 0
        i = bisect.bisect_right(ends, x) - 1
        if i < 0:
            if x <= intervals[0][1]:
                return x - intervals[0][0]
            else:
                return 0
        total = pre_sum[i+1]
        if i + 1 < m:
            next_s = intervals[i+1][0]
            if next_s <= x:
                total += x - next_s
        return total
    
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr+1])
        ptr += 2
        ans = compute(r) - compute(l)
        print(ans)

if __name__ == "__main__":
    main()