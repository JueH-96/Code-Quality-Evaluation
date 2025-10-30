from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    T = input[2]
    
    # Check if counts of B and W are the same in S and T
    if S.count('B') != T.count('B') or S.count('W') != T.count('W'):
        print(-1)
        return
    
    if S == T:
        print(0)
        return
    
    initial = S + '..'
    target = T + '..'
    
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
        
        if current_state == target:
            print(steps)
            found = True
            break
        
        # Find the empty positions (consecutive '.')
        empty_i = -1
        for i in range(len(current_state) - 1):
            if current_state[i] == '.' and current_state[i+1] == '.':
                empty_i = i
                break
        
        # Generate all possible moves
        for x in range(len(current_state) - 1):
            if x == empty_i:
                continue
            if current_state[x] != '.' and current_state[x+1] != '.':
                # Create new state
                new_state = list(current_state)
                # Move x and x+1 to empty_i and empty_i+1
                new_state[empty_i] = new_state[x]
                new_state[empty_i+1] = new_state[x+1]
                new_state[x] = '.'
                new_state[x+1] = '.'
                new_state_str = ''.join(new_state)
                
                if new_state_str not in visited:
                    visited.add(new_state_str)
                    queue.append((new_state_str, steps + 1))
    
    if not found:
        print(-1)

if __name__ == "__main__":
    main()