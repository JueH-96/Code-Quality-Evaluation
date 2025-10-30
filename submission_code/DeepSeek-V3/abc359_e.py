# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    A = [0] * (N + 1)
    operations = [0] * N
    total_operations = 0
    
    for i in range(N):
        # The number of operations needed for A[i] to become positive is:
        # sum_{j=1 to i} H[j] + 1
        # But since H[j] can be large, we need to compute it efficiently
        # The first time A[i] becomes positive is when A[i-1] > H[i] and A[i-1] > A[i]
        # Which happens after (sum_{j=1 to i} H[j]) + 1 operations
        # So, for each i, the answer is sum_{j=1 to i} H[j] + 1
        
        # To compute the sum efficiently, we can precompute the prefix sum
        if i == 0:
            operations[i] = H[i] + 1
        else:
            operations[i] = operations[i-1] + H[i]
    
    print(' '.join(map(str, operations)))

if __name__ == "__main__":
    main()