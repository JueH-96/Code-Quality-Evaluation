# Read input
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
K = int(input())

# Process each query
for _ in range(K):
    X, Y = map(int, input().split())
    
    # Calculate sum of absolute differences for current X, Y
    total = 0
    for i in range(X):
        for j in range(Y):
            total += abs(A[i] - B[j])
    
    # Print result for current query
    print(total)