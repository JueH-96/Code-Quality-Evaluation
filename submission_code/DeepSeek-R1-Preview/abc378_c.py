n = int(input())
a = list(map(int, input().split()))
last_occurrence = {}
b = []
for j in range(n):
    current = a[j]
    if current in last_occurrence:
        b.append(last_occurrence[current])
    else:
        b.append(-1)
    last_occurrence[current] = j + 1  # Using 1-based index
print(' '.join(map(str, b)))