def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Check if infinite is possible
    if sum(A) <= M:
        print("infinite")
        return
        
    # Binary search for maximum x
    left = 0
    right = max(A) + 1
    
    while right - left > 1:
        mid = (left + right) // 2
        total = sum(min(mid, a) for a in A)
        
        if total <= M:
            left = mid
        else:
            right = mid
            
    print(left)

solve()