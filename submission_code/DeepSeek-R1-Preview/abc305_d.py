import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    sleep_intervals = []
    for i in range(1, N, 2):
        if i + 1 >= len(A):
            break
        s = A[i]
        e = A[i+1]
        sleep_intervals.append((s, e))
    
    m = len(sleep_intervals)
    starts = []
    ends = []
    for s, e in sleep_intervals:
        starts.append(s)
        ends.append(e)
    
    prefix_sum = [0]
    current_sum = 0
    for s, e in sleep_intervals:
        current_sum += e - s
        prefix_sum.append(current_sum)
    
    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr+1])
        ptr += 2
        
        # Compute f(r)
        t = r
        j_end = bisect.bisect_right(ends, t) - 1
        if j_end >= 0:
            fr = prefix_sum[j_end + 1]
        else:
            fr = 0
        j_start = bisect.bisect_right(starts, t) - 1
        if j_start >= 0 and j_start > j_end:
            fr += t - starts[j_start]
        
        # Compute f(l)
        t = l
        j_end = bisect.bisect_right(ends, t) - 1
        if j_end >= 0:
            fl = prefix_sum[j_end + 1]
        else:
            fl = 0
        j_start = bisect.bisect_right(starts, t) - 1
        if j_start >= 0 and j_start > j_end:
            fl += t - starts[j_start]
        
        print(fr - fl)

if __name__ == '__main__':
    main()