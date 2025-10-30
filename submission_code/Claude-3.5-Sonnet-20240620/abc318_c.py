# YOUR CODE HERE
N, D, P = map(int, input().split())
F = list(map(int, input().split()))

F.sort(reverse=True)
total_cost = 0
i = 0

while i < N:
    batch_cost = sum(F[i:min(i+D, N)])
    if batch_cost > P:
        total_cost += P
        i += D
    else:
        total_cost += sum(F[i:])
        break

print(total_cost)