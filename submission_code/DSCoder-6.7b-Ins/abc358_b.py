N, A = map(int, input().split())
T = list(map(int, input().split()))

total_time = 0
for i in range(N):
    total_time += T[i] - total_time // A * A
    print(total_time + A)