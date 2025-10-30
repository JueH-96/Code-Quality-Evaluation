def can_form_pyramid(k, A):
    n = len(A)
    if 2 * k - 1 > n:
        return False
    
    # Check if we can form a pyramid of size k
    # We need to check if we can form the sequence 1, 2, ..., k, ..., 2, 1
    # within the bounds of the array A by decreasing values and removing edges
    required = list(range(1, k)) + list(range(k, 0, -1))
    l, r = 0, n - 1
    i = 0
    
    while i < len(required) and l <= r:
        if required[i] <= A[l]:
            l += 1
            i += 1
        elif required[i] <= A[r]:
            r -= 1
            i += 1
        else:
            return False
    
    return i == len(required)

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Binary search for the maximum k
    left, right = 1, (N // 2) + 1  # k ranges from 1 to at least N//2 + 1
    
    while left < right:
        mid = (left + right) // 2
        if can_form_pyramid(mid, A):
            left = mid + 1
        else:
            right = mid
    
    # The maximum k that worked is left-1 because left is the first that failed
    print(left - 1)