from collections import deque

def min_operations(N, X, A, B, P, Q):
    # Adjusting to 0-indexed
    X -= 1
    P = [p - 1 for p in P]
    Q = [q - 1 for q in Q]
    
    # Initial state: state[i] = (red_balls_in_box_i, blue_balls_in_box_i)
    initial_state = tuple([(A[i], B[i]) for i in range(N)])
    
    # Check if the initial state is already the goal state
    is_goal = True
    for i in range(N):
        if i != X and (initial_state[i][0] > 0 or initial_state[i][1] > 0):
            is_goal = False
            break
    if is_goal:
        return 0
    
    # Queue for BFS
    queue = deque([(initial_state, 0)])  # (state, operations_count)
    visited = {initial_state}
    
    while queue:
        state, ops = queue.popleft()
        
        for i in range(N):
            # Extract the balls from the i-th box
            red_balls, blue_balls = state[i]
            
            if red_balls == 0 and blue_balls == 0:
                continue  # No balls to move
            
            # Create a new state by moving balls
            new_state_list = list(state)
            new_state_list[i] = (0, 0)  # Empty the i-th box
            
            # Add red balls to P_i-th box
            new_state_list[P[i]] = (new_state_list[P[i]][0] + red_balls, new_state_list[P[i]][1])
            
            # Add blue balls to Q_i-th box
            new_state_list[Q[i]] = (new_state_list[Q[i]][0], new_state_list[Q[i]][1] + blue_balls)
            
            new_state = tuple(new_state_list)
            
            # Check if the new state is the goal state
            is_goal = True
            for j in range(N):
                if j != X and (new_state[j][0] > 0 or new_state[j][1] > 0):
                    is_goal = False
                    break
            if is_goal:
                return ops + 1
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, ops + 1))
    
    return -1  # Impossible to achieve the goal

def main():
    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    result = min_operations(N, X, A, B, P, Q)
    print(result)

if __name__ == "__main__":
    main()