import sys
data = sys.stdin.read().split()
N = int(data[0])
A = int(data[1])
T = [int(data[i]) for i in range(2, 2 + N)]

booth_free = 0
for t_arrive in T:
    start_time = max(t_arrive, booth_free)
    finish_time = start_time + A
    print(finish_time)
    booth_free = finish_time