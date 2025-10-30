import sys
import bisect

def main():
    data = sys.stdin.read().split()
    index = 0
    N = int(data[index])
    index += 1
    X = list(map(int, data[index:index+N]))
    index += N
    P = list(map(int, data[index:index+N]))
    index += N
    Q = int(data[index])
    index += 1
    queries = []
    for _ in range(Q):
        L = int(data[index])
        R = int(data[index+1])
        queries.append((L, R))
        index += 2

    # Compute prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + P[i]

    # Process each query
    results = []
    for L, R in queries:
        left = bisect.bisect_left(X, L)
        right = bisect.bisect_right(X, R) - 1
        if left <= right:
            total = prefix_sum[right + 1] - prefix_sum[left]
        else:
            total = 0
        results.append(str(total))
    
    # Output all results
    print('
'.join(results))

if __name__ == '__main__':
    main()