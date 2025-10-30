n = int(input())
a = list(map(int, input().split()))
result = []
for i in range(n):
    start = i * 7
    week = a[start:start+7]
    result.append(sum(week))
print(' '.join(map(str, result)))