def solve_for_position(slimes, start_pos):
    # Create a copy of slimes and track current position
    current_slimes = slimes[:]
    pos = start_pos
    current_size = current_slimes[pos]
    
    while True:
        absorbed = False
        
        # Check left neighbor
        left_can_absorb = pos > 0 and current_slimes[pos-1] < current_size
        # Check right neighbor  
        right_can_absorb = pos < len(current_slimes)-1 and current_slimes[pos+1] < current_size
        
        if left_can_absorb and right_can_absorb:
            # Choose the smaller one to absorb first
            if current_slimes[pos-1] <= current_slimes[pos+1]:
                # Absorb left
                current_size += current_slimes[pos-1]
                current_slimes.pop(pos-1)
                pos -= 1  # Position shifts left
                absorbed = True
            else:
                # Absorb right
                current_size += current_slimes[pos+1]
                current_slimes.pop(pos+1)
                absorbed = True
        elif left_can_absorb:
            # Absorb left
            current_size += current_slimes[pos-1]
            current_slimes.pop(pos-1)
            pos -= 1  # Position shifts left
            absorbed = True
        elif right_can_absorb:
            # Absorb right
            current_size += current_slimes[pos+1]
            current_slimes.pop(pos+1)
            absorbed = True
        
        if absorbed:
            current_slimes[pos] = current_size
        else:
            break
    
    return current_size

# Read input
n = int(input())
slimes = list(map(int, input().split()))

# Solve for each starting position
results = []
for k in range(n):
    max_size = solve_for_position(slimes, k)
    results.append(max_size)

# Output results
print(' '.join(map(str, results)))