def total_cost(N):
    cost = 0
    while N >= 2:
        cost += N
        N //= 2
    return cost

N = int(input().strip())
print(total_cost(N))