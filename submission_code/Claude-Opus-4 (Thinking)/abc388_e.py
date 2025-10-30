# YOUR CODE HERE
n = int(input())
A = list(map(int, input().split()))

def is_possible(k):
    if 2 * k > n:
        return False
    
    # Take the k smallest mochi as the top mochi
    # Try to pair each top mochi with a bottom mochi from the remaining
    
    j = k  # Start from the k-th mochi (0-indexed)
    for i in range(k):
        while j < n and 2 * A[i] > A[j]:
            j += 1
        if j == n:
            return False
        j += 1
    
    return True

# Binary search on the answer
left, right = 0, n // 2
while left < right:
    mid = (left + right + 1) // 2
    if is_possible(mid):
        left = mid
    else:
        right = mid - 1

print(left)