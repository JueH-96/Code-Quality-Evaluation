import sys
import bisect

def main():
    import sys
    import bisect
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr]); ptr += 1
    A = list(map(int, data[ptr:ptr+N])); ptr += N
    Q = int(data[ptr]); ptr += 1
    queries = []
    for _ in range(Q):
        l = int(data[ptr]); ptr += 1
        r = int(data[ptr]); ptr += 1
        queries.append((l, r))
    
    M = (N - 1) // 2
    periods = []
    for i in range(M):
        start = A[2*i + 1]
        end = A[2*i + 2]
        periods.append((start, end))
    
    start_times = [p[0] for p in periods]
    end_times = [p[1] for p in periods]
    
    prefix_sum = [0] * (M + 1)
    for i in range(M):
        prefix_sum[i+1] = prefix_sum[i] + (periods[i][1] - periods[i][0])
    
    answers = []
    for l, r in queries:
        left = bisect.bisect_left(start_times, l)
        right = bisect.bisect_right(end_times, r)
        
        total = prefix_sum[right] - prefix_sum[left]
        
        if left > 0:
            prev_end = periods[left-1][1]
            if prev_end > l:
                overlap_start = max(periods[left-1][0], l)
                overlap_end = min(prev_end, r)
                total += overlap_end - overlap_start
        if right < M:
            next_start = periods[right][0]
            if next_start < r:
                overlap_start = max(next_start, l)
                overlap_end = min(periods[right][1], r)
                total += overlap_end - overlap_start
        answers.append(str(total))
    
    print('
'.join(answers))

if __name__ == '__main__':
    main()