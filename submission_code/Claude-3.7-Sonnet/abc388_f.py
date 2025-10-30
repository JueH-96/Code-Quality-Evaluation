from collections import deque

def can_reach_end(n, m, a, b, bad_ranges):
    # If there are no bad ranges, the answer is "Yes"
    if m == 0:
        return True
    
    # Compute the good ranges from the bad ranges
    good_ranges = []
    
    # Add the first good range [1, L_1 - 1]
    good_ranges.append((1, bad_ranges[0][0] - 1))
    
    # Add the intermediate good ranges [R_i + 1, L_{i+1} - 1]
    for i in range(m - 1):
        good_ranges.append((bad_ranges[i][1] + 1, bad_ranges[i + 1][0] - 1))
    
    # Add the last good range [R_M + 1, N]
    good_ranges.append((bad_ranges[m - 1][1] + 1, n))
    
    # Check if we can navigate through all good ranges
    max_reachable = 1  # Start at square 1
    
    for i, (start, end) in enumerate(good_ranges):
        # If we can't even reach the start of the current good range, check if we can jump to it
        if max_reachable < start:
            if max_reachable + b < start:
                return False
        
        # Compute the entry point for the current good range
        entry_point = max(max_reachable, start)
        
        # Find the maximum reachable position within the current good range
        r_i = compute_max_reachable(start, end, a, b, entry_point)
        
        # Update the global maximum reachable position
        max_reachable = max(max_reachable, r_i)
        
        # If we've reached square N, the answer is "Yes"
        if max_reachable >= n:
            return True
    
    return max_reachable >= n

def compute_max_reachable(start, end, a, b, entry_point):
    # Initialize the queue for BFS
    queue = deque([entry_point])
    
    # Initialize the set of visited positions
    visited = set([entry_point])
    
    # Initialize the maximum reachable position
    max_reachable = entry_point
    
    # BFS
    while queue:
        curr = queue.popleft()
        
        # Try all possible jump distances
        for jump_dist in range(a, b + 1):
            new_pos = curr + jump_dist
            
            # Ensure the new position is within the good range and hasn't been visited
            if start <= new_pos <= end and new_pos not in visited:
                visited.add(new_pos)
                queue.append(new_pos)
                max_reachable = max(max_reachable, new_pos)
    
    return max_reachable

def main():
    # Read the input
    n, m, a, b = map(int, input().split())
    
    # Read the bad ranges
    bad_ranges = []
    for _ in range(m):
        l, r = map(int, input().split())
        bad_ranges.append((l, r))
    
    # Determine if we can reach square N
    if can_reach_end(n, m, a, b, bad_ranges):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()