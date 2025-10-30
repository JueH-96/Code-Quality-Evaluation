import collections

def solve():
    n, q = map(int, input().split())
    instructions = []
    for _ in range(q):
        h, t = input().split()
        instructions.append((h, int(t)))
    
    left_hand_pos = 1
    right_hand_pos = 2
    total_operations = 0
    
    for instruction in instructions:
        hand_to_move, target_part = instruction
        
        if hand_to_move == 'L':
            start_part = left_hand_pos
            fixed_part = right_hand_pos
            if start_part == target_part:
                operations = 0
            else:
                distance = -1
                distances = {i: float('inf') for i in range(1, n + 1)}
                distances[start_part] = 0
                queue = collections.deque([start_part])
                while queue:
                    current_part = queue.popleft()
                    if current_part == target_part:
                        distance = distances[target_part]
                        break
                    
                    neighbors = []
                    if current_part == 1:
                        neighbors = [n, 2]
                    elif current_part == n:
                        neighbors = [n - 1, 1]
                    else:
                        neighbors = [current_part - 1, current_part + 1]
                        
                    for neighbor in neighbors:
                        if neighbor != fixed_part and distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_part] + 1
                            queue.append(neighbor)
                operations = distance
            total_operations += operations
            left_hand_pos = target_part
            
        elif hand_to_move == 'R':
            start_part = right_hand_pos
            fixed_part = left_hand_pos
            if start_part == target_part:
                operations = 0
            else:
                distance = -1
                distances = {i: float('inf') for i in range(1, n + 1)}
                distances[start_part] = 0
                queue = collections.deque([start_part])
                while queue:
                    current_part = queue.popleft()
                    if current_part == target_part:
                        distance = distances[target_part]
                        break
                        
                    neighbors = []
                    if current_part == 1:
                        neighbors = [n, 2]
                    elif current_part == n:
                        neighbors = [n - 1, 1]
                    else:
                        neighbors = [current_part - 1, current_part + 1]
                        
                    for neighbor in neighbors:
                        if neighbor != fixed_part and distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_part] + 1
                            queue.append(neighbor)
                operations = distance
            total_operations += operations
            right_hand_pos = target_part
            
    print(total_operations)

if __name__ == '__main__':
    solve()