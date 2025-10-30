# YOUR CODE HERE
N, A = map(int, input().split())
T = list(map(int, input().split()))

current_time = 0
for i in range(N):
    if T[i] > current_time:
        current_time = T[i]
    current_time += A
    print(current_time)