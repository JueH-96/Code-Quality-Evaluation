n = int(input())
a = list(map(int, input().split()))

result = []
for i in range(n):
    week_sum = sum(a[i * 7 : (i + 1) * 7])
    result.append(week_sum)

print(*result)