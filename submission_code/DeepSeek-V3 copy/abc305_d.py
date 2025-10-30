def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    
    A = list(map(int, data[idx:idx+N]))
    idx += N
    
    Q = int(data[idx])
    idx += 1
    
    queries = []
    for _ in range(Q):
        l = int(data[idx])
        r = int(data[idx+1])
        queries.append((l, r))
        idx += 2
    
    # Precompute the total sleep time up to each A_i
    total_sleep = [0] * N
    for i in range(1, N, 2):
        total_sleep[i] = total_sleep[i-1] + (A[i+1] - A[i])
    for i in range(2, N, 2):
        total_sleep[i] = total_sleep[i-1]
    
    # Function to find the index of the first element >= x
    def lower_bound(x):
        low = 0
        high = N - 1
        while low <= high:
            mid = (low + high) // 2
            if A[mid] < x:
                low = mid + 1
            else:
                high = mid - 1
        return low
    
    for l, r in queries:
        # Find the first A_i >= l
        start = lower_bound(l)
        # Find the first A_i >= r
        end = lower_bound(r)
        
        # Calculate the total sleep time between start and end
        if start >= N:
            print(0)
            continue
        if end >= N:
            end = N - 1
        
        # Initialize the total sleep time
        total = 0
        
        # Iterate through the sleep sessions
        for i in range(start, end):
            if i % 2 == 1:
                # Sleep session: A[i] to A[i+1]
                sleep_start = max(A[i], l)
                sleep_end = min(A[i+1], r)
                if sleep_start < sleep_end:
                    total += sleep_end - sleep_start
        
        print(total)

if __name__ == "__main__":
    main()