# YOUR CODE HERE
def can_see(buildings, start, end):
    """Check if building at index 'end' can be seen from building at index 'start'"""
    target_height = buildings[end]
    for k in range(start + 1, end):
        if buildings[k] > target_height:
            return False
    return True

n, q = map(int, input().split())
heights = list(map(int, input().split()))

for _ in range(q):
    l, r = map(int, input().split())
    l -= 1  # Convert to 0-based indexing
    r -= 1  # Convert to 0-based indexing
    
    count = 0
    # Check buildings to the east of building r (r+1 to n-1)
    for j in range(r + 1, n):
        # Check if building j can be seen from both building l and building r
        if can_see(heights, l, j) and can_see(heights, r, j):
            count += 1
    
    print(count)