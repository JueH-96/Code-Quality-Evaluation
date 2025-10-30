# YOUR CODE HERE
def can_fit(words, W, M):
    """Check if we can fit all words in at most M lines with window width W"""
    if len(words) == 0:
        return True
    
    lines = 1
    current_width = words[0]
    
    for i in range(1, len(words)):
        # Try to add word i to current line
        # Need to add 1 for space + word width
        if current_width + 1 + words[i] <= W:
            current_width += 1 + words[i]
        else:
            # Start new line
            lines += 1
            if lines > M:
                return False
            current_width = words[i]
    
    return True

def solve():
    N, M = map(int, input().split())
    L = list(map(int, input().split()))
    
    # Binary search on window width
    # Minimum possible width is the maximum word length
    # Maximum possible width is sum of all words + spaces
    left = max(L)
    right = sum(L) + (N - 1)
    
    while left < right:
        mid = (left + right) // 2
        if can_fit(L, mid, M):
            right = mid
        else:
            left = mid + 1
    
    print(left)

solve()