def solve():
    N = int(input())
    
    water_amount = 0  # Initialize water amount to 0
    current_time = 0  # Initialize current time to 0
    
    for _ in range(N):
        T, V = map(int, input().split())
        
        # Calculate water lost due to leakage
        time_elapsed = T - current_time
        water_amount = max(0, water_amount - time_elapsed)
        
        # Add new water
        water_amount += V
        
        # Update current time
        current_time = T
    
    return water_amount

print(solve())