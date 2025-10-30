import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # Check if B and W counts are equal
    if S.count('B') != T.count('B') or S.count('W') != T.count('W'):
        print(-1)
        return
    
    if S == T:
        print(0)
        return
    
    # Initialize BFS
    start_state = S + '..'
    target_state = T + '..'
    visited = {start_state: 0}
    queue = deque([start_state])
    
    len_state = n + 2  # Length of each state string
    
    while queue:
        current = queue.popleft()
        if current == target_state:
            print(visited[current])
            return
        
        steps = visited[current]
        # Find the position of '..'
        k = current.find('..')
        
        # Generate all possible moves
        for x in range(len_state - 1):
            if current[x] != '.' and current[x+1] != '.':
                # Create new_state
                new_state = list(current)
                # Move x and x+1 to k and k+1
                new_state[k] = new_state[x]
                new_state[k+1] = new_state[x+1]
                new_state[x] = '.'
                new_state[x+1] = '.'
                new_state_str = ''.join(new_state)
                
                if new_state_str not in visited:
                    visited[new_state_str] = steps + 1
                    queue.append(new_state_str)
    
    # If target not reached
    print(-1)

if __name__ == "__main__":
    main()