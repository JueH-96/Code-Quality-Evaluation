def minimum_movements(N, positions, Q, tasks):
    total_movements = 0
    current_positions = positions[:]
    
    for T, G in tasks:
        T_index = T - 1  # Convert to 0-based index
        current_position = current_positions[T_index]
        
        # Calculate the movements required to move person T to G
        movements = abs(current_position - G)
        total_movements += movements
        
        # Update the position of the T-th person
        current_positions[T_index] = G
        
        # After moving T to G, we need to ensure no other person occupies G
        # We need to check if G is already occupied by another person
        for i in range(N):
            if i != T_index and current_positions[i] == G:
                # If occupied, we need to move the other person
                # Move the other person to the nearest free position
                if current_positions[i] < G:
                    # Move left to the nearest free position
                    while current_positions[i] == G:
                        current_positions[i] -= 1
                        total_movements += 1
                else:
                    # Move right to the nearest free position
                    while current_positions[i] == G:
                        current_positions[i] += 1
                        total_movements += 1
    
    return total_movements

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
positions = list(map(int, data[1].split()))
Q = int(data[2])
tasks = [tuple(map(int, line.split())) for line in data[3:3 + Q]]

result = minimum_movements(N, positions, Q, tasks)
print(result)