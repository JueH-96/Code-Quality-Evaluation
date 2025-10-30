n = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

x_max = float('inf')
for i in range(n):
    if A[i] > 0:
        current = Q[i] // A[i]
        if current < x_max:
            x_max = current

max_total = 0

for x in range(x_max + 1):
    y_max = float('inf')
    for i in range(n):
        if B[i] == 0:
            continue
        rem = Q[i] - x * A[i]
        y_candidate = rem // B[i]
        if y_candidate < y_max:
            y_max = y_candidate
    total = x + y_max
    if total > max_total:
        max_total = total

print(max_total)