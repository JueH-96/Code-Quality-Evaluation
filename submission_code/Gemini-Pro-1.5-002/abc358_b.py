# YOUR CODE HERE
n, a = map(int, input().split())
t = list(map(int, input().split()))

finish_times = []
current_time = 0

for arrival_time in t:
    start_time = max(arrival_time, current_time)
    finish_time = start_time + a
    finish_times.append(finish_time)
    current_time = finish_time

for finish_time in finish_times:
    print(finish_time)