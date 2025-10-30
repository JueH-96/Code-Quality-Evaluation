n = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Compute the maximum possible x (servings of A)
x_max_candidates = [Q[i] // A[i] for i in range(n) if A[i] > 0]
x_max = min(x_max_candidates) if x_max_candidates else 0

max_total = 0

for x in range(x_max + 1):
    remaining = [Q[i] - A[i] * x for i in range(n)]
    y_candidates = []
    for i in range(n):
        if B[i] > 0:
            y = remaining[i] // B[i]
            y_candidates.append(y)
    if y_candidates:
        y_max = min(y_candidates)
        current_total = x + y_max
        if current_total > max_total:
            max_total = current_total

print(max_total)