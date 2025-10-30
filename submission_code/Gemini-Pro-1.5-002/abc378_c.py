n = int(input())
a = list(map(int, input().split()))
b = []
last_occurrence = {}
for i in range(n):
    if a[i] in last_occurrence:
        b.append(last_occurrence[a[i]])
    else:
        b.append(-1)
    last_occurrence[a[i]] = i + 1
print(*b)