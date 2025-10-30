import math

def solve():
    n, q = map(int, input().split())
    instructions = []
    for _ in range(q):
        h, t = input().split()
        instructions.append((h, int(t)))
    
    left_hand_pos = 1
    right_hand_pos = 2
    total_operations = 0
    
    def get_neighbors(part, n_parts):
        if part == 1:
            return [n_parts, 2]
        elif part == n_parts:
            return [n_parts - 1, 1]
        else:
            return [part - 1, part + 1]
            
    def get_shortest_path_ops(start_pos, target_pos, fixed_pos):
        if start_pos == target_pos:
            return 0
        
        queue = [((start_pos, fixed_pos), 0)]
        visited_states = set([(start_pos, fixed_pos)])
        
        while queue:
            current_state, ops_count = queue.pop(0)
            current_hand_pos, other_hand_pos = current_state[0], current_state[1]
            
            if current_hand_pos == target_pos:
                return ops_count
                
            neighbors = get_neighbors(current_hand_pos, n)
            for next_hand_pos in neighbors:
                if next_hand_pos != other_hand_pos:
                    next_state = (next_hand_pos, other_hand_pos)
                    if next_state not in visited_states:
                        visited_states.add(next_state)
                        queue.append((next_state, ops_count + 1))
                        
        return float('inf') # Should not reach here as solution is guaranteed to exist
        
    for instruction in instructions:
        hand_instruction, target_part = instruction
        if hand_instruction == 'R':
            start_right_pos = right_hand_pos
            target_right_pos = target_part
            fixed_left_pos = left_hand_pos
            
            ops = get_shortest_path_ops(start_right_pos, target_right_pos, fixed_left_pos)
            total_operations += ops
            right_hand_pos = target_right_pos
            
        elif hand_instruction == 'L':
            start_left_pos = left_hand_pos
            target_left_pos = target_part
            fixed_right_pos = right_hand_pos
            
            ops = get_shortest_path_ops(start_left_pos, target_left_pos, fixed_right_pos)
            total_operations += ops
            left_hand_pos = target_left_pos
            
    print(total_operations)

if __name__ == '__main__':
    solve()