n, c = map(int, input().split())
t = list(map(int, input().split()))

candies = 0
last_candy_time = -c

for time in t:
    if time - last_candy_time >= c:
        candies += 1
        last_candy_time = time

print(candies)