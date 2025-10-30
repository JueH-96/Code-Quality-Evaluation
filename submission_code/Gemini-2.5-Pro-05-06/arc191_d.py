import collections
import sys

def main():
    N, M, S_in, T_in = map(int, sys.stdin.readline().split())
    S = S_in - 1  # 0-indexed
    T = T_in - 1  # 0-indexed

    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # 0-indexed
        v -= 1 # 0-indexed
        adj[u].append(v)
        adj[v].append(u)
    
    # Initialize queue for BFS. Stores tuples of (current_A_pos, current_B_pos).
    queue = collections.deque()
    
    # visited dictionary stores minimum distance to a state (pos_A, pos_B).
    # Key: (pos_A, pos_B) tuple, Value: integer distance.
    visited = {} 
    
    # Initial state: A at S, B at T. Distance is 0.
    initial_state = (S, T)
    queue.append(initial_state)
    visited[initial_state] = 0
    
    min_ops = -1

    while queue:
        curr_A, curr_B = queue.popleft()
        dist = visited[(curr_A, curr_B)]

        # Check if target state (A at T, B at S) is reached
        if curr_A == T and curr_B == S:
            min_ops = dist
            break
        
        # Try to move piece A
        for next_A in adj[curr_A]:
            if next_A == curr_B:  # Collision: A cannot move to where B is
                continue
            
            new_state = (next_A, curr_B)
            if new_state not in visited: # If not visited, this is the shortest path to new_state
                visited[new_state] = dist + 1
                queue.append(new_state)

        # Try to move piece B
        for next_B in adj[curr_B]:
            if next_B == curr_A:  # Collision: B cannot move to where A is
                continue
            
            new_state = (curr_A, next_B)
            if new_state not in visited: # If not visited, this is the shortest path to new_state
                visited[new_state] = dist + 1
                queue.append(new_state)
                
    sys.stdout.write(str(min_ops) + "
")

if __name__ == '__main__':
    main()