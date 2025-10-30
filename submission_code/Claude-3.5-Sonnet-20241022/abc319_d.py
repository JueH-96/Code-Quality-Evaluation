def can_fit(N, M, L, width):
    lines = 1
    curr_width = L[0]
    
    for i in range(1, N):
        # Try to add word to current line
        if curr_width + 1 + L[i] <= width:
            curr_width += 1 + L[i]
        else:
            # Need new line
            lines += 1
            curr_width = L[i]
            if lines > M:
                return False
    return True

def solve():
    N, M = map(int, input().split())
    L = list(map(int, input().split()))
    
    # Binary search on window width
    left = max(L)  # Minimum possible width is max word length
    right = sum(L) + N-1  # Maximum width is all words on one line
    
    while left < right:
        mid = (left + right) // 2
        if can_fit(N, M, L, mid):
            right = mid
        else:
            left = mid + 1
            
    print(left)

solve()