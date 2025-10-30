def solve():
    # Read input
    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    P = list(map(lambda x: int(x)-1, input().split()))  # Convert to 0-based indexing
    Q = list(map(lambda x: int(x)-1, input().split()))
    X -= 1  # Convert to 0-based indexing
    
    # Create state representation - tuple of (box_num, is_red)
    balls = []
    for i in range(N):
        if A[i] == 1:
            balls.append((i, True))
        if B[i] == 1:
            balls.append((i, False))
    
    # BFS to find shortest path
    from collections import deque, defaultdict
    
    def state_to_tuple(balls):
        return tuple(sorted(balls))
    
    def is_goal(state):
        boxes = set(box for box, _ in state)
        return len(boxes - {X}) == 0
    
    def get_next_states(state):
        boxes = defaultdict(list)
        for box, is_red in state:
            boxes[box].append(is_red)
            
        next_states = []
        for box in range(N):
            if not boxes[box]:
                continue
            
            # Move all balls from this box
            new_state = []
            for other_box, is_red in state:
                if other_box != box:
                    new_state.append((other_box, is_red))
            
            # Put red balls in P[box], blue balls in Q[box]
            for is_red in boxes[box]:
                if is_red:
                    new_state.append((P[box], True))
                else:
                    new_state.append((Q[box], False))
                    
            next_states.append(state_to_tuple(new_state))
        return next_states
    
    # Run BFS
    start_state = state_to_tuple(balls)
    if is_goal(start_state):
        return 0
        
    visited = {start_state}
    queue = deque([(start_state, 0)])
    
    while queue:
        state, moves = queue.popleft()
        
        for next_state in get_next_states(state):
            if next_state in visited:
                continue
                
            if is_goal(next_state):
                return moves + 1
                
            visited.add(next_state)
            queue.append((next_state, moves + 1))
    
    return -1

print(solve())