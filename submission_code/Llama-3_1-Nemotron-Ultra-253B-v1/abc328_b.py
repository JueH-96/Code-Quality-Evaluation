n = int(input())
d = list(map(int, input().split()))
count = 0
for i in range(1, n + 1):
    days = d[i - 1]
    for j in range(1, days + 1):
        s = str(i) + str(j)
        if all(c == s[0] for c in s):
            count += 1
print(count)