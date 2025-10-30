# YOUR CODE HERE
def ticket_purchasing(N, A, T):
    finish_times = []
    current_time = 0

    for i in range(N):
        arrival_time = T[i]
        if current_time < arrival_time:
            current_time = arrival_time
        current_time += A
        finish_times.append(current_time)

    return finish_times

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = int(data[1])
T = list(map(int, data[2:]))

results = ticket_purchasing(N, A, T)
for result in results:
    print(result)