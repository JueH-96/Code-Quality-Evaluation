def solve_slime():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    # For each starting position K
    result = []
    for k in range(N):
        # Make a copy of the array and simulate
        slimes = A.copy()
        pos = k  # Current position of Takahashi
        size = slimes[k]  # Current size of Takahashi
        
        while True:
            # Try to find an adjacent smaller slime
            can_absorb = False
            
            # Check left neighbor if exists
            if pos > 0 and slimes[pos-1] < size:
                # Absorb left
                size += slimes[pos-1]
                slimes.pop(pos-1)
                pos -= 1
                can_absorb = True
                continue
                
            # Check right neighbor if exists
            if pos < len(slimes)-1 and slimes[pos+1] < size:
                # Absorb right
                size += slimes[pos+1]
                slimes.pop(pos+1)
                can_absorb = True
                continue
                
            # If no absorption possible, we're done
            if not can_absorb:
                break
                
        result.append(size)
    
    # Print results
    print(*result)

# Run the solution
solve_slime()