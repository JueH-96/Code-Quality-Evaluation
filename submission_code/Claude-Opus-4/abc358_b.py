# YOUR CODE HERE
N, A = map(int, input().split())
T = list(map(int, input().split()))

# Track when the booth becomes free
booth_free_time = 0

for i in range(N):
    arrival_time = T[i]
    
    # Person starts purchasing either when they arrive or when booth becomes free
    start_time = max(arrival_time, booth_free_time)
    
    # Person finishes A seconds after they start
    finish_time = start_time + A
    
    # Update when booth becomes free
    booth_free_time = finish_time
    
    # Output the finish time
    print(finish_time)