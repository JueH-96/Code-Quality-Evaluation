n, a = map(int, input().split())
t = list(map(int, input().split()))

booth_free_time = 0
for i in range(n):
    arrival_time = t[i]
    start_time = max(arrival_time, booth_free_time)
    finish_time = start_time + a
    booth_free_time = finish_time
    print(finish_time)