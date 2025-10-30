import collections

def solve():
    n = int(input())
    s_str = input()
    t_str = input()
    s_counts = {'B': 0, 'W': 0}
    t_counts = {'B': 0, 'W': 0}
    for char in s_str:
        s_counts[char] += 1
    for char in t_str:
        t_counts[char] += 1
    if s_counts['B'] != t_counts['B'] or s_counts['W'] != t_counts['W']:
        print("-1")
        return
        
    initial_state = tuple(list(s_str) + ['.', '.'])
    target_state = tuple(list(t_str) + ['.', '.'])
    
    if initial_state == target_state:
        print(0)
        return
        
    queue = collections.deque([(initial_state, 0)])
    visited_states = {initial_state}
    
    while queue:
        current_state, operations_count = queue.popleft()
        
        empty_indices = []
        for i in range(n + 2):
            if current_state[i] == '.':
                empty_indices.append(i)
        if len(empty_indices) < 2:
            continue
        e1, e2 = empty_indices[0], empty_indices[1]
        
        for x in range(n + 1):
            if current_state[x] != '.' and current_state[x+1] != '.':
                v_x = current_state[x]
                v_x_plus_1 = current_state[x+1]
                
                next_state_list = list(current_state)
                next_state_list[e1] = v_x
                next_state_list[e2] = v_x_plus_1
                next_state_list[x] = '.'
                next_state_list[x+1] = '.'
                next_state = tuple(next_state_list)
                
                if next_state not in visited_states:
                    if next_state == target_state:
                        print(operations_count + 1)
                        return
                    visited_states.add(next_state)
                    queue.append((next_state, operations_count + 1))
                    
    print("-1")

if __name__ == '__main__':
    solve()