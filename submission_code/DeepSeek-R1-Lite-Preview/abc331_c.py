import sys
import bisect

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    sorted_A = sorted(A)
    
    # Compute prefix sums from the end
    prefix_sum = [0] * N
    prefix_sum[-1] = sorted_A[-1]
    for i in range(N-2, -1, -1):
        prefix_sum[i] = sorted_A[i] + prefix_sum[i+1]
    
    result = []
    for a in A:
        idx = bisect.bisect_right(sorted_A, a)
        if idx < N:
            result.append(str(prefix_sum[idx]))
        else:
            result.append('0')
    
    print(' '.join(result))

if __name__ == '__main__':
    main()