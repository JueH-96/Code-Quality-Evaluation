# YOUR CODE HERE
def solve():
    N = int(input())
    
    current_water = 0
    previous_time = 0

    for _ in range(N):
        line = input().split()
        T_i = int(line[0])
        V_i = int(line[1])
        
        # Calculate time elapsed since the previous event
        time_elapsed = T_i - previous_time
        
        # Water leaks during this interval.
        # current_water becomes the level just before addition at T_i.
        current_water = max(0, current_water - time_elapsed)
        
        # Add V_i liters of water at T_i.
        # current_water becomes the level immediately after addition at T_i.
        current_water += V_i
        
        # Update previous_time for the next iteration.
        previous_time = T_i
        
    print(current_water)

if __name__ == '__main__':
    solve()
# YOUR CODE HERE