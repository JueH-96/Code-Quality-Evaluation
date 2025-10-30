# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = list(map(int, data[1:N+1]))
    P = list(map(int, data[N+1:2*N+1]))
    Q = int(data[2*N+1])
    queries = [list(map(int, data[2*N+2+2*i:2*N+4+2*i])) for i in range(Q)]
    
    # Create a prefix sum array for P
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + P[i]
    
    # Process each query
    results = []
    for L, R in queries:
        left = bisect_left(X, L)
        right = bisect_right(X, R)
        result = prefix_sum[right] - prefix_sum[left]
        results.append(result)
    
    for result in results:
        print(result)

def bisect_left(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo

def bisect_right(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] > x:
            hi = mid
        else:
            lo = mid + 1
    return lo

if __name__ == "__main__":
    main()