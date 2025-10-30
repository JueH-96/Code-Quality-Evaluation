import collections

def solve():
    n = int(input())
    s_str = input()
    t_str = input()
    s_counts = {'B': 0, 'W': 0}
    t_counts = {'B': 0, 'W': 0}
    for char_s in s_str:
        s_counts[char_s] += 1
    for char_t in t_str:
        t_counts[char_t] += 1
    if s_counts['B'] != t_counts['B'] or s_counts['W'] != t_counts['W']:
        print("-1")
        return
    
    initial_state = list(s_str + "..")
    target_state = list(t_str + "..")
    
    start_state_str = "".join(initial_state)
    target_state_str = "".join(target_state)
    
    if start_state_str == target_state_str:
        print(0)
        return
        
    queue = collections.deque([(start_state_str, 0)])
    visited_states = {start_state_str}
    
    while queue:
        current_state_str, operations_count = queue.popleft()
        if current_state_str == target_state_str:
            print(operations_count)
            return
            
        current_state = list(current_state_str)
        empty_indices = []
        for i in range(n + 2):
            if current_state[i] == '.':
                empty_indices.append(i)
        if len(empty_indices) != 2:
            continue 
        empty_pos1, empty_pos2 = min(empty_indices), max(empty_indices)
        
        for x in range(n + 1):
            if current_state[x] != '.' and current_state[x+1] != '.':
                next_state = list(current_state)
                stone1 = next_state[x]
                stone2 = next_state[x+1]
                next_state[empty_pos1] = stone1
                next_state[empty_pos2] = stone2
                next_state[x] = '.'
                next_state[x+1] = '.'
                next_state_str = "".join(next_state)
                if next_state_str not in visited_states:
                    visited_states.add(next_state_str)
                    queue.append((next_state_str, operations_count + 1))
                    
    print("-1")

if __name__ == '__main__':
    solve()