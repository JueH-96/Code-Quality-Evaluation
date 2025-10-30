def solve():
    # Read input
    N, C = map(int, input().split())
    T = list(map(int, input().split()))
    
    # Initialize variables
    candies = 0
    last_candy_time = -float('inf')  # Set to negative infinity to ensure the first press gives a candy
    
    # Process each button press
    for time in T:
        if time - last_candy_time >= C:
            # If enough time has elapsed, give a candy
            candies += 1
            last_candy_time = time
    
    return candies

# Read input and print solution
print(solve())