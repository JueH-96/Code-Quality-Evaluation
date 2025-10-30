n = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the maximum possible x (number of dish A servings)
x_max = min(Q[i] // A[i] for i in range(n) if A[i] > 0)

max_total = 0

for x in range(x_max + 1):
    y_list = []
    for i in range(n):
        if B[i] > 0:
            available = Q[i] - x * A[i]
            y_i = available // B[i]
            y_list.append(y_i)
    if y_list:
        y = min(y_list)
        total = x + y
        if total > max_total:
            max_total = total

print(max_total)