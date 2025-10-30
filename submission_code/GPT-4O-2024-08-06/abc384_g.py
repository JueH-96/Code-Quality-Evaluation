# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    # Read N
    N = int(data[index])
    index += 1
    
    # Read A
    A = list(map(int, data[index:index + N]))
    index += N
    
    # Read B
    B = list(map(int, data[index:index + N]))
    index += N
    
    # Read K
    K = int(data[index])
    index += 1
    
    results = []
    
    # Process each query
    for _ in range(K):
        X_k = int(data[index])
        Y_k = int(data[index + 1])
        index += 2
        
        # Calculate the sum of absolute differences
        total_sum = 0
        for i in range(X_k):
            for j in range(Y_k):
                total_sum += abs(A[i] - B[j])
        
        results.append(total_sum)
    
    # Print results
    for result in results:
        print(result)

main()