import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    R = list(map(int, input[ptr:ptr+N]))
    ptr += N
    R.sort()
    
    # Compute prefix sums
    pre_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        pre_sum[i] = pre_sum[i-1] + R[i-1]
    
    # Process each query
    for _ in range(Q):
        X = int(input[ptr])
        ptr += 1
        k = bisect.bisect_right(pre_sum, X) - 1
        print(k)

if __name__ == '__main__':
    main()