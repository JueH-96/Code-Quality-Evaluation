N, A = map(int, input().split())
T = list(map(int, input().split()))

finish_time = 0
for i in range(N):
    if T[i] > finish_time:
        finish_time += A - (T[i] - finish_time)
    finish_time += A
    print(finish_time)