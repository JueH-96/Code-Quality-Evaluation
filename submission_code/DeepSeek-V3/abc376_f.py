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
            target = T
            current = left
            other = right
        else:
            target = T
            current = right
            other = left
        
        # Calculate the minimal steps
        # Since it's a ring, we can go clockwise or counter-clockwise
        # The minimal steps is min(abs(current - target), N - abs(current - target))
        # But we need to ensure that the other hand is not on the target
        # So, we need to find a path where the other hand is not on the target
        
        # First, calculate the minimal steps without considering the other hand
        steps = min(abs(current - target), N - abs(current - target))
        
        # Check if the other hand is on the target
        if other == target:
            # Need to move the other hand first
            # Find the nearest position to move the other hand
            # We can move it to either current-1 or current+1, but not to target
            # Since it's a ring, we need to handle wrap-around
            # Let's try moving it to current-1
            new_other = other - 1
            if new_other < 1:
                new_other = N
            # Check if new_other is not equal to target
            if new_other == target:
                # Try moving to current+1
                new_other = other + 1
                if new_other > N:
                    new_other = 1
            # Now, move the other hand to new_other
            # Calculate the steps to move the other hand
            steps_other = min(abs(other - new_other), N - abs(other - new_other))
            # Then, move the current hand to target
            steps_current = min(abs(current - target), N - abs(current - target))
            total_steps = steps_other + steps_current
        else:
            total_steps = steps
        
        total_ops += total_steps
        
        # Update the positions
        if H == 'L':
            left = target
        else:
            right = target
    
    print(total_ops)

if __name__ == "__main__":
    main()