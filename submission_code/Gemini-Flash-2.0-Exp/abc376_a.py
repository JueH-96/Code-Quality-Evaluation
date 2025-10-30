N, C = map(int, input().split())
T = list(map(int, input().split()))

candies = 0
last_candy_time = -float('inf')

for time in T:
  if time - last_candy_time >= C:
    candies += 1
    last_candy_time = time

print(candies)