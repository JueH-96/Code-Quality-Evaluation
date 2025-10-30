from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    T = data[2]
    
    # Check if S and T have the same number of 'W's and 'B's
    if S.count('W') != T.count('W') or S.count('B') != T.count('B'):
        print(-1)
        return
    
    # Initial and target states
    initial = S + '..'
    target = T + '..'
    
    if initial == target:
        print(0)
        return
    
    visited = set()
    visited.add(initial)
    queue = deque()
    queue.append((initial, 0))
    
    while queue:
        current, steps = queue.popleft()
        
        # Find positions of the two empty cells (should be adjacent)
        e = current.find('.')
        if e == -1 or current[e+1] != '.':
            continue  # Invalid state
        k = e  # index of first empty cell
        k1 = e + 1  # index of second empty cell
        
        # Try moving stones from each possible pair of adjacent cells
        for x in range(N+1):  # x from 0 to N (0-indexed)
            if current[x] != '.' and current[x+1] != '.':
                # Create a list for easy manipulation
                list_state = list(current)
                # Move stones to empty cells
                list_state[k] = list_state[x]
                list_state[k1] = list_state[x+1]
                # Set moved cells to empty
                list_state[x] = '.'
                list_state[x+1] = '.'
                new_state = ''.join(list_state)
                if new_state == target:
                    print(steps + 1)
                    return
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, steps + 1))
    
    # If target state is not reached
    print(-1)

if __name__ == "__main__":
    main()