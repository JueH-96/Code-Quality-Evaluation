def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    X = list(map(int, data[1].split()))
    P = list(map(int, data[2].split()))
    Q = int(data[3])
    
    queries = []
    for i in range(4, 4 + Q):
        L, R = map(int, data[i].split())
        queries.append((L, R))
    
    # Precompute prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + P[i - 1]
    
    results = []
    
    for L, R in queries:
        # Binary search for the left and right bounds
        from bisect import bisect_left, bisect_right
        
        left_index = bisect_left(X, L)
        right_index = bisect_right(X, R) - 1
        
        if left_index <= right_index:
            total_villagers = prefix_sum[right_index + 1] - prefix_sum[left_index]
        else:
            total_villagers = 0
        
        results.append(total_villagers)
    
    # Print all results
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()