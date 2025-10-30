N, M = map(int, input().split())
A = list(map(int, input().split()))

# Check if we can pay everyone their full transportation cost
total_full_cost = sum(A)
if total_full_cost <= M:
    print("infinite")
else:
    # Binary search for the maximum valid x
    left, right = 0, max(A)
    
    while left < right:
        mid = (left + right + 1) // 2
        total_cost = sum(min(mid, a) for a in A)
        
        if total_cost <= M:
            left = mid
        else:
            right = mid - 1
    
    print(left)