from collections import deque

def solve():
    # Read input
    N, M, A, B = map(int, input().split())
    bad_intervals = []
    for _ in range(M):
        L, R = map(int, input().split())
        bad_intervals.append((L, R))
    
    # BFS with range tracking
    # Each element in queue is a tuple (start, end) representing a range of positions
    queue = deque([(1, 1)])  # Start at position 1
    visited_ranges = set()
    
    while queue:
        start, end = queue.popleft()
        
        # Skip if we've already processed this range
        if (start, end) in visited_ranges:
            continue
        visited_ranges.add((start, end))
        
        # Check if we can reach N from this range
        if start <= N <= end:
            print("Yes")
            return
        
        # Generate all possible next positions from this range
        # From any position x in [start, end], we can reach positions in [x+A, x+B]
        # So from the range [start, end], we can reach [start+A, end+B]
        next_start = start + A
        next_end = min(end + B, N)
        
        if next_start > N:
            continue
        
        # Now we need to remove bad squares from [next_start, next_end]
        ranges_to_add = [(next_start, next_end)]
        
        # Remove bad intervals
        for L, R in bad_intervals:
            new_ranges = []
            for rs, re in ranges_to_add:
                if re < L or rs > R:
                    # No intersection
                    new_ranges.append((rs, re))
                else:
                    # There's intersection, split the range
                    if rs < L:
                        new_ranges.append((rs, L - 1))
                    if re > R:
                        new_ranges.append((R + 1, re))
            ranges_to_add = new_ranges
        
        # Add valid ranges to queue
        for rs, re in ranges_to_add:
            if rs <= re:
                queue.append((rs, re))
    
    print("No")

solve()