from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    s = input[1]
    t = input[2]
    
    initial = s + '..'
    if initial[:n] == t:
        print(0)
        return
    
    visited = set()
    queue = deque()
    queue.append((initial, 0))
    visited.add(initial)
    
    found = False
    
    while queue:
        current_state, steps = queue.popleft()
        
        # Check if the first n characters match T
        if current_state[:n] == t:
            print(steps)
            found = True
            break
        
        # Find the position of the empty cells
        empty_pos = -1
        for i in range(len(current_state) - 1):
            if current_state[i] == '.' and current_state[i+1] == '.':
                empty_pos = i
                break
        
        # Generate all possible next states
        for x in range(len(current_state) - 1):
            if current_state[x] != '.' and current_state[x+1] != '.':
                # Create new state
                new_state = list(current_state)
                new_state[empty_pos] = new_state[x]
                new_state[empty_pos + 1] = new_state[x + 1]
                new_state[x] = '.'
                new_state[x + 1] = '.'
                new_state_str = ''.join(new_state)
                
                if new_state_str not in visited:
                    visited.add(new_state_str)
                    queue.append((new_state_str, steps + 1))
    
    if not found:
        print(-1)

if __name__ == "__main__":
    main()