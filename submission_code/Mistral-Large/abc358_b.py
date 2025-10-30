import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
T = list(map(int, data[2:]))

current_time = 0
finish_times = []

for i in range(N):
    if current_time < T[i]:
        current_time = T[i] + A
    else:
        current_time += A
    finish_times.append(current_time)

for time in finish_times:
    print(time)