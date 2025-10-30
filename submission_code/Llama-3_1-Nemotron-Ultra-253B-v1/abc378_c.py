n = int(input())
a = list(map(int, input().split()))
last_occurrence = {}
b = []
for i in range(n):
    current = a[i]
    if current in last_occurrence:
        b.append(last_occurrence[current] + 1)
    else:
        b.append(-1)
    last_occurrence[current] = i
print(' '.join(map(str, b)))