# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # Read N
    index = 0
    N = int(data[index])
    index += 1
    
    # Read A
    A = list(map(int, data[index:index + N]))
    index += N
    
    # Read Q
    Q = int(data[index])
    index += 1
    
    # Read queries
    queries = []
    for _ in range(Q):
        l = int(data[index])
        r = int(data[index + 1])
        queries.append((l, r))
        index += 2
    
    # Precompute sleep time using prefix sum
    max_time = A[-1]
    sleep_time = [0] * (max_time + 1)
    
    # Fill sleep_time array
    for i in range(1, N // 2 + 1):
        start_sleep = A[2 * i - 1]
        end_sleep = A[2 * i]
        for t in range(start_sleep, end_sleep):
            sleep_time[t] = 1
    
    # Compute prefix sum
    for t in range(1, max_time + 1):
        sleep_time[t] += sleep_time[t - 1]
    
    # Answer each query
    results = []
    for l, r in queries:
        results.append(sleep_time[r] - sleep_time[l])
    
    # Print results
    for result in results:
        print(result)