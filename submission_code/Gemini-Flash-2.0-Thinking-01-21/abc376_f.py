import collections

def solve():
    n, q = map(int, input().split())
    instructions = []
    for _ in range(q):
        h, t = input().split()
        instructions.append((h, int(t)))
    
    left_pos = 1
    right_pos = 2
    total_operations = 0
    
    for instruction in instructions:
        hand_type, target_part = instruction
        start_state = (left_pos, right_pos)
        target_reached = False
        queue = collections.deque([(start_state, 0)])
        visited_states = {start_state}
        min_ops_for_instruction = -1
        final_state = None
        
        while queue:
            current_state, operations_count = queue.popleft()
            current_left, current_right = current_state
            
            if hand_type == 'L' and current_left == target_part:
                target_reached = True
                min_ops_for_instruction = operations_count
                final_state = current_state
                break
            elif hand_type == 'R' and current_right == target_part:
                target_reached = True
                min_ops_for_instruction = operations_count
                final_state = current_state
                break
                
            # Possible moves for left hand
            next_l_pos = (current_left % n) + 1
            if next_l_pos != current_right:
                next_state = (next_l_pos, current_right)
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    queue.append((next_state, operations_count + 1))
                    
            prev_l_pos = ((current_left - 2 + n) % n) + 1
            if prev_l_pos != current_right:
                next_state = (prev_l_pos, current_right)
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    queue.append((next_state, operations_count + 1))
                    
            # Possible moves for right hand
            next_r_pos = (current_right % n) + 1
            if next_r_pos != current_left:
                next_state = (current_left, next_r_pos)
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    queue.append((next_state, operations_count + 1))
                    
            prev_r_pos = ((current_right - 2 + n) % n) + 1
            if prev_r_pos != current_left:
                next_state = (current_left, prev_r_pos)
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    queue.append((next_state, operations_count + 1))
                    
        if target_reached:
            total_operations += min_ops_for_instruction
            left_pos, right_pos = final_state
        else:
            # This should not happen based on problem description
            pass 
            
    print(total_operations)

if __name__ == '__main__':
    solve()