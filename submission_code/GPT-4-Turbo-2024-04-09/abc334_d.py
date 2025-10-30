import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    R = []
    for i in range(N):
        R.append(int(data[index]))
        index += 1
    
    queries = []
    for i in range(Q):
        queries.append(int(data[index]))
        index += 1
    
    # Sort the requirements for sleighs
    R.sort()
    
    # Precompute the prefix sums of R
    prefix_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + R[i - 1]
    
    # Answer each query
    results = []
    for X in queries:
        # Binary search for the maximum number of sleighs that can be pulled
        low, high = 0, N
        while low < high:
            mid = (low + high + 1) // 2
            if prefix_sum[mid] <= X:
                low = mid
            else:
                high = mid - 1
        results.append(low)
    
    # Print results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()