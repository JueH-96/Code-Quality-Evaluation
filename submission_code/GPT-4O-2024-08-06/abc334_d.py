# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # Read N and Q
    index = 0
    N = int(data[index])
    Q = int(data[index + 1])
    index += 2
    
    # Read R_1, R_2, ..., R_N
    R = list(map(int, data[index:index + N]))
    index += N
    
    # Read queries
    queries = list(map(int, data[index:index + Q]))
    
    # Sort the reindeer requirements
    R.sort()
    
    # Compute prefix sums
    prefix_sums = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sums[i] = prefix_sums[i - 1] + R[i - 1]
    
    # Function to find the maximum number of sleighs that can be pulled with X reindeer
    def max_sleighs(X):
        # Binary search on the prefix sums
        low, high = 0, N
        while low < high:
            mid = (low + high + 1) // 2
            if prefix_sums[mid] <= X:
                low = mid
            else:
                high = mid - 1
        return low
    
    # Process each query
    results = []
    for X in queries:
        results.append(max_sleighs(X))
    
    # Print results
    for result in results:
        print(result)