# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    
    queries = []
    index = 3
    for _ in range(Q):
        l = int(data[index])
        r = int(data[index + 1])
        queries.append((l, r))
        index += 2
    
    # Preprocess the string S to find consecutive duplicates
    consecutive_counts = [0] * (N - 1)
    for i in range(N - 1):
        if S[i] == S[i + 1]:
            consecutive_counts[i] = 1
    
    # Create a prefix sum array for consecutive_counts
    prefix_sum = [0] * (N)
    for i in range(1, N):
        prefix_sum[i] = prefix_sum[i - 1] + consecutive_counts[i - 1]
    
    # Answer each query
    results = []
    for l, r in queries:
        # Convert 1-based index to 0-based index
        l -= 1
        r -= 1
        # Calculate the number of consecutive duplicates in the range [l, r]
        result = prefix_sum[r] - prefix_sum[l]
        results.append(result)
    
    # Print the results
    for res in results:
        print(res)

main()