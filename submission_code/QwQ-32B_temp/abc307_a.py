n = int(input())
a = list(map(int, input().split()))
result = []
for i in range(n):
    start = i * 7
    week_sum = sum(a[start:start+7])
    result.append(str(week_sum))
print(' '.join(result))