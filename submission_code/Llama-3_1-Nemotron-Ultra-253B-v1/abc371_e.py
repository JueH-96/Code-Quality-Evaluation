n = int(input())
a = list(map(int, input().split()))
next_occurrence = [n] * n
last_occurrence = dict()
for i in range(n-1, -1, -1):
    val = a[i]
    next_occurrence[i] = last_occurrence.get(val, n)
    last_occurrence[val] = i
total = 0
for i in range(n):
    total += (i + 1) * (next_occurrence[i] - i)
print(total)