# YOUR CODE HERE
def max_pyramid_size(N, A):
    left = 1
    right = N
    max_k = 0
    
    while left <= right:
        mid = (left + right) // 2
        possible = True
        for i in range(mid):
            if A[i] < i + 1:
                possible = False
                break
        for i in range(mid, 2 * mid - 1):
            if A[i] < (2 * mid - 1 - i):
                possible = False
                break
        if possible:
            max_k = mid
            left = mid + 1
        else:
            right = mid - 1
    return max_k

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute the maximum pyramid size
result = max_pyramid_size(N, A)

# Print the result
print(result)