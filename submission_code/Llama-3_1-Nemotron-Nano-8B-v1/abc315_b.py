m = int(input())
d = list(map(int, input().split()))
total = sum(d)
target = (total + 1) // 2
cum = 0
for i in range(m):
    cum += d[i]
    if cum >= target:
        a = i + 1
        b = target - (cum - d[i])
        print(a, b)
        break