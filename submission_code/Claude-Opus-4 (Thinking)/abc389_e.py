n, m = map(int, input().split())
prices = list(map(int, input().split()))

# Binary search on the threshold
left, right = 0, 2 * 10**10  # Upper bound on threshold

while left < right:
    mid = (left + right + 1) // 2
    total_cost = 0
    for p in prices:
        k = (mid + p) // (2 * p)
        total_cost += k * k * p
    if total_cost <= m:
        left = mid
    else:
        right = mid - 1

threshold = left

# Calculate units and cost with the found threshold
total_units = 0
total_cost = 0
units_bought = []
for p in prices:
    k = (threshold + p) // (2 * p)
    total_units += k
    total_cost += k * k * p
    units_bought.append(k)

# Find the next units to buy with remaining budget
remaining_budget = m - total_cost
next_units = []
for i in range(n):
    k = units_bought[i]
    marginal_cost = (2 * k + 1) * prices[i]
    next_units.append((marginal_cost, i))

# Sort by marginal cost
next_units.sort()

# Buy as many as possible
for mc, i in next_units:
    if mc <= remaining_budget:
        remaining_budget -= mc
        total_units += 1

print(total_units)