# YOUR CODE HERE
def max_gifts(N, M, A):
    A.sort()
    left = 0
    max_count = 0
    for right in range(N):
        while A[right] - A[left] >= M:
            left += 1
        max_count = max(max_count, right - left + 1)
    return max_count

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Compute and print the result
print(max_gifts(N, M, A))