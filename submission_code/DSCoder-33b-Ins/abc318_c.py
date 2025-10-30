# YOUR CODE HERE

N, D, P = map(int, input().split())
F = list(map(int, input().split()))

# Sort the regular fare in ascending order
F.sort()

# Calculate the minimum total cost
total_cost = sum(F[:N % D]) * P + sum(F[N % D:])

print(total_cost)