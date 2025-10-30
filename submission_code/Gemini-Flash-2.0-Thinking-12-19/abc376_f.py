import collections

def solve():
    n, q = map(int, input().split())
    instructions = []
    for _ in range(q):
        h, t = input().split()
        instructions.append({'hand': h, 'target_part': int(t)})
    
    current_left_hand_pos = 1
    current_right_hand_pos = 2
    total_operations = 0
    
    for instruction in instructions:
        target_hand = instruction['hand']
        target_part = instruction['target_part']
        
        if target_hand == 'L':
            start_state = (current_left_hand_pos, current_right_hand_pos)
            target_states = set([(target_part, r) for r in range(1, n + 1) if r != target_part])
        else: # target_hand == 'R'
            start_state = (current_left_hand_pos, current_right_hand_pos)
            target_states = set([(l, target_part) for l in range(1, n + 1) if l != target_part])
            
        if (target_hand == 'L' and current_left_hand_pos == target_part) or (target_hand == 'R' and current_right_hand_pos == target_part):
            operations_for_instruction = 0
            next_state = start_state
        else:
            queue = collections.deque([(start_state, 0)])
            visited_states = {start_state}
            path_found = False
            operations_for_instruction = -1
            next_state = None
            
            while queue:
                current_state, operations_count = queue.popleft()
                l_pos, r_pos = current_state
                
                if current_state in target_states:
                    operations_for_instruction = operations_count
                    next_state = current_state
                    path_found = True
                    break
                    
                # Move left hand
                for next_l_pos in [((l_pos - 2 + n) % n) + 1, (l_pos % n) + 1]:
                    if next_l_pos != r_pos:
                        next_state_tuple = (next_l_pos, r_pos)
                        if next_state_tuple not in visited_states:
                            visited_states.add(next_state_tuple)
                            queue.append((next_state_tuple, operations_count + 1))
                            
                # Move right hand
                for next_r_pos in [((r_pos - 2 + n) % n) + 1, (r_pos % n) + 1]:
                    if next_r_pos != l_pos:
                        next_state_tuple = (l_pos, next_r_pos)
                        if next_state_tuple not in visited_states:
                            visited_states.add(next_state_tuple)
                            queue.append((next_state_tuple, operations_count + 1))
                            
            if not path_found:
                raise Exception("Path not found, which should not happen based on problem description")
                
        total_operations += operations_for_instruction
        current_left_hand_pos = next_state[0]
        current_right_hand_pos = next_state[1]
        
    print(total_operations)

if __name__ == '__main__':
    solve()