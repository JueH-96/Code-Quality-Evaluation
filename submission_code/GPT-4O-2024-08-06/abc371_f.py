# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    # Number of persons
    N = int(data[index])
    index += 1
    
    # Initial positions of persons
    X = list(map(int, data[index:index + N]))
    index += N
    
    # Number of tasks
    Q = int(data[index])
    index += 1
    
    # Tasks
    tasks = []
    for _ in range(Q):
        T = int(data[index]) - 1  # Convert to 0-based index
        G = int(data[index + 1])
        tasks.append((T, G))
        index += 2
    
    # Current positions of persons
    current_positions = X[:]
    occupied_positions = set(X)
    
    total_movement = 0
    
    for T, G in tasks:
        current_pos = current_positions[T]
        
        # Calculate the movement cost
        movement_cost = abs(current_pos - G)
        total_movement += movement_cost
        
        # Move the person to the target position
        occupied_positions.remove(current_pos)
        
        # Find the nearest unoccupied position to G
        if G not in occupied_positions:
            new_pos = G
        else:
            # If G is occupied, find the nearest unoccupied position
            left = G - 1
            right = G + 1
            while True:
                if left not in occupied_positions:
                    new_pos = left
                    break
                if right not in occupied_positions:
                    new_pos = right
                    break
                left -= 1
                right += 1
        
        # Update the person's position
        current_positions[T] = new_pos
        occupied_positions.add(new_pos)
    
    print(total_movement)

if __name__ == "__main__":
    main()