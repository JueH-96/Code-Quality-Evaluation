N, C = map(int, input().split())
T = list(map(int, input().split()))

last_time = 0
candies = 0
for i in range(N):
    if T[i] - last_time >= C:
        candies += 1
        last_time = T[i]

print(candies)