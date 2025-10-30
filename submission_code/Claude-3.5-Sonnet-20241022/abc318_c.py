N, D, P = map(int, input().split())
F = list(map(int, input().split()))

F.sort(reverse=True)
total = sum(F)
min_cost = total

# Try using 0,1,2,... batches of passes
for i in range(1, (N + D - 1) // D + 1):
    # Cost with i batches of passes
    cost = i * P
    # Add regular fares for remaining days after using passes
    for j in range(i * D, N):
        cost += F[j]
    min_cost = min(min_cost, cost)

print(min_cost)