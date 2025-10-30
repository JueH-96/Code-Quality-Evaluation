n = int(input())
a = list(map(int, input().split()))
last_occurrence = {}
b = []
for i in range(1, n + 1):
    current = a[i - 1]
    if current in last_occurrence:
        b.append(str(last_occurrence[current]))
    else:
        b.append(str(-1))
    last_occurrence[current] = i
print(' '.join(b))