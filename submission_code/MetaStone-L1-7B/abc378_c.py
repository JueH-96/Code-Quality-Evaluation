n = int(input())
a = list(map(int, input().split()))
last_occurrence = {}
B = [-1] * (n + 1)

for i in range(1, n + 1):
    current = a[i - 1]
    if current in last_occurrence:
        B[i] = last_occurrence[current]
    else:
        B[i] = -1
    last_occurrence[current] = i

print(' '.join(map(str, B[1:n+1])))