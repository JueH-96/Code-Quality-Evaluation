def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    current_L = 1
    current_R = 2
    total_steps = 0
    
    for _ in range(Q):
        H = input[ptr]
        ptr += 1
        T = int(input[ptr])
        ptr += 1
        
        if H == 'L':
            prev = current_L
            other_prev = current_R
        else:
            prev = current_R
            other_prev = current_L
        
        # Calculate minimal steps in both directions
        d_clockwise = (T - prev) % N
        d_counter_clockwise = (prev - T) % N
        minimal_steps = min(d_clockwise, d_counter_clockwise)
        
        # Check if other hand is at the target position
        if other_prev == T:
            steps_to_add = minimal_steps + 1
        else:
            steps_to_add = minimal_steps
        
        total_steps += steps_to_add
        
        # Update current position
        if H == 'L':
            current_L = T
        else:
            current_R = T
    
    print(total_steps)

if __name__ == "__main__":
    main()