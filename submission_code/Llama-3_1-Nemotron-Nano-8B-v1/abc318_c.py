n, d, p = map(int, input().split())
f = list(map(int, input().split()))
f.sort(reverse=True)

suffix = [0] * (n + 1)
for i in range(n-1, -1, -1):
    suffix[i] = suffix[i+1] + f[i]

min_cost = suffix[0]  # Case when no passes are used

for m in range(1, n + 1):
    batches = (m + d - 1) // d
    current_cost = batches * p + suffix[m]
    if current_cost < min_cost:
        min_cost = current_cost

print(min_cost)