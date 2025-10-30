# YOUR CODE HERE
N = int(input())
bases = []
for _ in range(N):
    W, X = map(int, input().split())
    bases.append((W, X))

bases.sort(key=lambda x: x[1])

total_employees = 0
for W, X in bases:
    if X + 1 < 24:
        total_employees += W
    else:
        break

print(total_employees)