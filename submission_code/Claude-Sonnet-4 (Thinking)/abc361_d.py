from collections import deque

n = int(input())
s = input().strip()
t = input().strip()

# Check if transformation is possible
if sorted(s) != sorted(t):
    print(-1)
    exit()

if s == t:
    print(0)
    exit()

# Initial state
initial = s + '..'

# BFS
queue = deque([(initial, 0)])
visited = {initial}

while queue:
    current, ops = queue.popleft()
    
    if current[:n] == t:
        print(ops)
        exit()
    
    # Find the position of the two consecutive empty cells
    empty_start = -1
    for i in range(len(current) - 1):
        if current[i] == '.' and current[i+1] == '.':
            empty_start = i
            break
    
    # Try all possible moves
    for x in range(len(current) - 1):
        if current[x] != '.' and current[x+1] != '.':
            # Move stones from positions x, x+1 to empty positions
            new_state = list(current)
            new_state[empty_start] = current[x]
            new_state[empty_start + 1] = current[x+1]
            new_state[x] = '.'
            new_state[x+1] = '.'
            
            new_state_str = ''.join(new_state)
            if new_state_str not in visited:
                visited.add(new_state_str)
                queue.append((new_state_str, ops + 1))

print(-1)