def solve():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))

    # Compute left_max: maximum sum achievable by merging to the left
    left_max = [0] * N
    left_max[0] = A[0]
    for i in range(1, N):
        if A[i] > left_max[i-1]:
            left_max[i] = left_max[i-1] + A[i]
        else:
            left_max[i] = A[i]
    
    # Compute right_max: maximum sum achievable by merging to the right
    right_max = [0] * N
    right_max[-1] = A[-1]
    for i in range(N-2, -1, -1):
        if A[i] > right_max[i+1]:
            right_max[i] = right_max[i+1] + A[i]
        else:
            right_max[i] = A[i]
    
    # Calculate the result for each K
    result = []
    for i in range(N):
        res = left_max[i] + right_max[i] - A[i]
        result.append(str(res))
    
    print(' '.join(result))

solve()