import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()
    
    # Check if the number of W's are the same
    if s.count('W') != t.count('W'):
        print(-1)
        return
    
    initial = tuple(s + '..')
    target = tuple(t + '..')
    
    if initial == target:
        print(0)
        return
    
    visited = set()
    queue = deque()
    queue.append((initial, 0))
    visited.add(initial)
    
    found = False
    
    while queue:
        current_state, steps = queue.popleft()
        
        # Find the position of the empty cells
        e = -1
        for i in range(len(current_state) - 1):
            if current_state[i] == '.' and current_state[i+1] == '.':
                e = i
                break
        
        # Generate all possible moves
        for x in range(len(current_state) - 1):
            if current_state[x] != '.' and current_state[x+1] != '.':
                # Create new state
                new_state_list = list(current_state)
                new_state_list[e] = current_state[x]
                new_state_list[e+1] = current_state[x+1]
                new_state_list[x] = '.'
                new_state_list[x+1] = '.'
                new_state = tuple(new_state_list)
                
                if new_state == target:
                    print(steps + 1)
                    found = True
                    return
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))
    
    print(-1)

if __name__ == "__main__":
    main()