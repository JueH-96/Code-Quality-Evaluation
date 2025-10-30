import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
T = list(map(int, data[2:]))

queue = []
time = 0

for i in range(N):
    while queue and queue[0][0] <= T[i]:
        time = max(time, queue[0][1])
        queue.pop(0)
    
    if queue:
        time += A
    else:
        time = T[i]
    
    queue.append((T[i], time + A))

for t in queue:
    print(t[1])