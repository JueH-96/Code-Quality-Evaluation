# YOUR CODE HERE
n, a = map(int, input().split())
t = list(map(int, input().split()))

booth_free_time = 0

for i in range(n):
    arrival_time = t[i]
    
    # When does this person start purchasing?
    start_time = max(arrival_time, booth_free_time)
    
    # When do they finish?
    finish_time = start_time + a
    
    # Update when the booth becomes free
    booth_free_time = finish_time
    
    print(finish_time)