# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Check if we can make x infinitely large
if sum(A) <= M:
    print("infinite")
else:
    # Binary search for the maximum x
    left, right = 0, max(A)
    
    while left < right:
        mid = (left + right + 1) // 2
        total = sum(min(mid, a) for a in A)
        
        if total <= M:
            left = mid
        else:
            right = mid - 1
    
    print(left)