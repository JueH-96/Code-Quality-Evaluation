n = int(input())
a = list(map(int, input().split()))
last_occurrence = {}
b = [0] * n

for i in range(n):
    current = a[i]
    if current in last_occurrence:
        b[i] = last_occurrence[current]
    else:
        b[i] = -1
    last_occurrence[current] = i + 1

print(' '.join(map(str, b)))