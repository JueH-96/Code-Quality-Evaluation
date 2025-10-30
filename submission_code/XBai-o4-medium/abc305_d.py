import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + N]))
    ptr += N
    m = (N - 1) // 2
    starts = [A[2 * i + 1] for i in range(m)]
    ends = [A[2 * i + 2] for i in range(m)]
    
    # Compute prefix_sum
    prefix_sum = [0] * (m + 1)
    for i in range(1, m + 1):
        prefix_sum[i] = prefix_sum[i-1] + (ends[i-1] - starts[i-1])
    
    Q = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(Q):
        l = int(input[ptr])
        r = int(input[ptr + 1])
        ptr += 2
        
        # Compute f(r)
        x = r
        k = bisect.bisect_right(ends, x)
        total_r = prefix_sum[k]
        if k < m:
            if starts[k] <= x:
                total_r += x - starts[k]
        
        # Compute f(l)
        x = l
        k = bisect.bisect_right(ends, x)
        total_l = prefix_sum[k]
        if k < m:
            if starts[k] <= x:
                total_l += x - starts[k]
        
        results.append(str(total_r - total_l))
    
    print('
'.join(results))

if __name__ == "__main__":
    main()