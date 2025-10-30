# YOUR CODE HERE
n, a = map(int, input().split())
t = list(map(int, input().split()))

booth_free_at = 0

for i in range(n):
    # Person i arrives at t[i]
    # They start purchasing at the later of their arrival time or when booth is free
    start_time = max(t[i], booth_free_at)
    finish_time = start_time + a
    booth_free_at = finish_time
    print(finish_time)