def can_see_building(heights, from_pos, target_pos):
    # Check if building at target_pos can be seen from building at from_pos
    max_height = max(heights[from_pos+1:target_pos], default=-1)
    return heights[target_pos] > max_height

def solve():
    # Read input
    N, Q = map(int, input().split())
    H = list(map(int, input().split()))
    
    # Process each query
    for _ in range(Q):
        l, r = map(int, input().split())
        l -= 1  # Convert to 0-based indexing
        r -= 1
        
        count = 0
        # Check each building to the east of r
        for j in range(r + 1, N):
            # For a building j to be counted, it must be visible from both l and r
            if can_see_building(H, l, j) and can_see_building(H, r, j):
                count += 1
        
        print(count)

# Run the solution
solve()