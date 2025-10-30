n = int(input())
a = list(map(int, input().split()))
b = []
for i in range(n):
    start = i * 7
    end = start + 7
    week = a[start:end]
    b.append(sum(week))
print(' '.join(map(str, b)))