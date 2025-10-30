N, M, A, B = map(int, input().split())

bad_intervals = []
for _ in range(M):
    L, R = map(int, input().split())
    bad_intervals.append((L, R))

pos = 1

for L, R in bad_intervals:
    # If we're already past this interval, continue
    if pos > R:
        continue
    
    # If we're before the interval, try to jump over it
    if pos < L:
        # Find the furthest we can reach before entering the bad interval
        max_reach = pos
        while max_reach + A < L:
            max_reach += B
        
        # From max_reach, can we jump over the bad interval?
        if max_reach + A > R:
            pos = max_reach + B
            continue
    
    # If we're in the interval or can't jump over it, check more carefully
    # Use BFS to find if we can get past this interval
    from collections import deque
    
    queue = deque([pos])
    visited = set([pos])
    found = False
    
    while queue and not found:
        curr = queue.popleft()
        
        if curr > R:
            pos = curr
            found = True
            break
            
        for jump in range(A, B + 1):
            next_pos = curr + jump
            if next_pos > N:
                continue
            if L <= next_pos <= R:  # Bad square
                continue
            if next_pos in visited:
                continue
                
            visited.add(next_pos)
            queue.append(next_pos)
            
            if next_pos > R:
                pos = next_pos
                found = True
                break
    
    if not found:
        print("No")
        exit()

# Check if we can reach N
if pos + B >= N:
    print("Yes")
else:
    print("No")