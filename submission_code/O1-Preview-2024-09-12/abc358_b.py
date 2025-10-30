# YOUR CODE HERE
N, A = map(int, input().split())
T = list(map(int, input().split()))
current_time = 0
for t in T:
    if t >= current_time:
        finish_time = t + A
    else:
        finish_time = current_time + A
    print(finish_time)
    current_time = finish_time