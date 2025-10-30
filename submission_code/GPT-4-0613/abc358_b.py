N, A = map(int, input().split())
T = list(map(int, input().split()))

time = 0
for i in range(N):
    if T[i] > time:
        time = T[i]
    print(time + A)
    time += A