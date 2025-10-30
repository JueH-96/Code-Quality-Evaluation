# YOUR CODE HERE
N, A = map(int, input().split())
T = list(map(int, input().split()))
current_time = 0
for i in range(N):
    arrival_time = T[i]
    if arrival_time > current_time:
        current_time = arrival_time
    finish_time = current_time + A
    print(finish_time)
    current_time = finish_time