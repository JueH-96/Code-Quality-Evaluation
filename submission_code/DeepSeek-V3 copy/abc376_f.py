# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    instructions = []
    index = 2
    for _ in range(Q):
        H = data[index]
        T = int(data[index+1])
        instructions.append((H, T))
        index += 2
    
    # Initialize positions
    left = 1
    right = 2
    
    total_ops = 0
    
    for H, T in instructions:
        if H == 'L':
            current = left
            other = right
        else:
            current = right
            other = left
        
        if current == T:
            continue
        
        # Calculate the minimal steps
        # Since it's a ring, we can move clockwise or counter-clockwise
        # We need to ensure that the other hand is not on the destination
        # So, we need to find a path that avoids the other hand
        
        # Calculate the distance in both directions
        # Clockwise: (T - current) % N
        # Counter-clockwise: (current - T) % N
        
        # But since it's a ring, the minimal steps is min((T - current) % N, (current - T) % N)
        # However, we need to ensure that the other hand is not on the destination
        
        # So, we need to find a path that avoids the other hand
        
        # Let's try to move in the direction that minimizes the steps and avoids the other hand
        
        # First, calculate the minimal steps without considering the other hand
        steps_clockwise = (T - current) % N
        steps_counter_clockwise = (current - T) % N
        
        # Choose the direction with the minimal steps
        if steps_clockwise < steps_counter_clockwise:
            direction = 1  # clockwise
            steps = steps_clockwise
        else:
            direction = -1  # counter-clockwise
            steps = steps_counter_clockwise
        
        # Now, check if the destination is the other hand
        if (current + direction * steps) % N == other % N:
            # Need to choose the other direction
            if direction == 1:
                direction = -1
                steps = steps_counter_clockwise
            else:
                direction = 1
                steps = steps_clockwise
        
        # Now, perform the steps
        if direction == 1:
            # Move clockwise
            for _ in range(steps):
                current = (current + 1) % N
                if current == 0:
                    current = N
                total_ops += 1
        else:
            # Move counter-clockwise
            for _ in range(steps):
                current = (current - 1) % N
                if current == 0:
                    current = N
                total_ops += 1
        
        # Update the position
        if H == 'L':
            left = T
        else:
            right = T
    
    print(total_ops)

if __name__ == "__main__":
    main()