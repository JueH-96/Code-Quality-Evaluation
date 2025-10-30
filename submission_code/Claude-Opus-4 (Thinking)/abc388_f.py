# YOUR CODE HERE
def solve():
    # Read input
    N, M, A, B = map(int, input().split())
    bad_intervals = []
    for _ in range(M):
        L, R = map(int, input().split())
        bad_intervals.append((L, R))
    
    # Special case: no bad intervals
    if M == 0:
        print("Yes")
        return
    
    # Compute good segments
    good_segments = []
    
    # First segment: [1, L_1-1]
    if bad_intervals[0][0] > 1:
        good_segments.append((1, bad_intervals[0][0] - 1))
    
    # Middle segments
    for i in range(M - 1):
        if bad_intervals[i][1] + 1 <= bad_intervals[i+1][0] - 1:
            good_segments.append((bad_intervals[i][1] + 1, bad_intervals[i+1][0] - 1))
    
    # Last segment: [R_M+1, N]
    if bad_intervals[-1][1] < N:
        good_segments.append((bad_intervals[-1][1] + 1, N))
    
    # Find which segment contains 1 and N
    start_segment = -1
    target_segment = -1
    for i, (l, r) in enumerate(good_segments):
        if l <= 1 <= r:
            start_segment = i
        if l <= N <= r:
            target_segment = i
    
    if start_segment == -1 or target_segment == -1:
        print("No")
        return
    
    if start_segment == target_segment:
        print("Yes")
        return
    
    # Build adjacency list
    n_segments = len(good_segments)
    adj = [[] for _ in range(n_segments)]
    
    for i in range(n_segments):
        l_i, r_i = good_segments[i]
        for j in range(n_segments):
            if i == j:
                continue
            l_j, r_j = good_segments[j]
            
            # Check if we can reach segment j from segment i
            # From segment [l_i, r_i], we can reach positions [l_i + A, r_i + B]
            reach_start = l_i + A
            reach_end = r_i + B
            
            # Check if this range intersects with segment j
            if reach_start <= r_j and reach_end >= l_j:
                adj[i].append(j)
    
    # BFS from start_segment to target_segment
    from collections import deque
    queue = deque([start_segment])
    visited = [False] * n_segments
    visited[start_segment] = True
    
    while queue:
        curr = queue.popleft()
        if curr == target_segment:
            print("Yes")
            return
        
        for next_seg in adj[curr]:
            if not visited[next_seg]:
                visited[next_seg] = True
                queue.append(next_seg)
    
    print("No")

solve()