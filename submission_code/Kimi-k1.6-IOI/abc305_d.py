import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr+1])
        queries.append((l, r))
        ptr += 2
    
    # Preprocess sleep intervals
    sleep_intervals = []
    for i in range(1, N, 2):
        sleep_intervals.append((A[i], A[i+1]))
    
    ends = [e for s, e in sleep_intervals]
    
    # Precompute prefix sums
    prefix_sum = [0]
    current = 0
    for s, e in sleep_intervals:
        current += e - s
        prefix_sum.append(current)
    
    results = []
    for l, r in queries:
        # Compute cumulative sleep time up to r
        k_r = bisect.bisect_right(ends, r) - 1
        cum_r = prefix_sum[k_r + 1]
        if k_r + 1 < len(sleep_intervals):
            s, e = sleep_intervals[k_r + 1]
            if s <= r:
                cum_r += min(e, r) - s
        
        # Compute cumulative sleep time up to l
        k_l = bisect.bisect_right(ends, l) - 1
        cum_l = prefix_sum[k_l + 1]
        if k_l + 1 < len(sleep_intervals):
            s, e = sleep_intervals[k_l + 1]
            if s <= l:
                cum_l += min(e, l) - s
        
        results.append(cum_r - cum_l)
    
    print('
'.join(map(str, results)))

if __name__ == '__main__':
    main()