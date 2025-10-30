n, c = map(int, input().split())
t = list(map(int, input().split()))
count = 1
last = t[0]
for i in range(1, n):
    diff = t[i] - last
    if diff >= c:
        count += 1
        last = t[i]
print(count)