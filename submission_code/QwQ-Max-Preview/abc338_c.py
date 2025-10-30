n = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Compute x_max as the minimum Q[i]//A[i] for all i where A[i] > 0
x_max_candidates = [Q[i] // A[i] for i in range(n) if A[i] > 0]
x_max = min(x_max_candidates) if x_max_candidates else 0

max_total = 0

for x in range(x_max + 1):
    y_candidates = []
    for i in range(n):
        if B[i] == 0:
            continue
        rem = Q[i] - A[i] * x
        y_i = rem // B[i]
        y_candidates.append(y_i)
    if not y_candidates:
        y = 0
    else:
        y = min(y_candidates)
    current_total = x + y
    if current_total > max_total:
        max_total = current_total

print(max_total)