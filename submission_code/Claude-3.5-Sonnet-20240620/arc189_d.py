# YOUR CODE HERE
def solve(N, A):
    stack = []
    right_max = [0] * N
    
    # Calculate the maximum size for each slime looking to the right
    for i in range(N - 1, -1, -1):
        while stack and A[stack[-1]] < A[i]:
            right_max[stack.pop()] = A[i]
        stack.append(i)
    
    # Calculate the maximum size for each slime
    result = [0] * N
    stack = []
    for i in range(N):
        while stack and A[stack[-1]] < A[i]:
            j = stack.pop()
            result[j] = max(result[j], A[i])
        if stack:
            result[i] = max(result[i], A[stack[-1]])
        result[i] = max(result[i], A[i], right_max[i])
        stack.append(i)
    
    return result

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and print output
result = solve(N, A)
print(*result)