n = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

valid_A = [i for i in range(n) if A[i] > 0]
x_max = min(Q[i] // A[i] for i in valid_A) if valid_A else 0

max_total = 0

for x in range(x_max + 1):
    possible_y = []
    for i in range(n):
        if B[i] > 0:
            numerator = Q[i] - x * A[i]
            possible_y.append(numerator // B[i])
    if not possible_y:
        continue  # should not happen per problem constraints
    y = min(possible_y)
    current_total = x + y
    if current_total > max_total:
        max_total = current_total

print(max_total)