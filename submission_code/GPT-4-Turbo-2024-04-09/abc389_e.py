import sys
import math

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
P = list(map(int, data[2:]))

# To maximize the number of units, we should focus on products with the smallest P_i first
# because they cost less per unit squared.

# Sort products by their price factor P_i
sorted_indices = sorted(range(N), key=lambda x: P[x])

# We will use binary search to find the maximum number of units we can buy
def can_buy_units(total_units):
    total_cost = 0
    for index in sorted_indices:
        # Calculate the maximum units we can buy for this product
        # k^2 * P[index] <= M
        # k^2 <= M / P[index]
        # k <= sqrt(M / P[index])
        max_units_for_product = int(math.sqrt(M / P[index]))
        
        # We can only buy up to `total_units` in total across all products
        if total_units <= max_units_for_product:
            # If the remaining total_units is less than what we can buy for this product
            cost = total_units * total_units * P[index]
        else:
            # Otherwise, we buy as much as we can for this product
            cost = max_units_for_product * max_units_for_product * P[index]
            total_units -= max_units_for_product
        
        total_cost += cost
        
        if total_cost > M:
            return False
        
        if total_units == 0:
            break
    
    return total_cost <= M

# Binary search for the maximum number of units
low, high = 0, 10**18  # Start with an arbitrarily large high, we'll adjust it in the loop
while low < high:
    mid = (low + high + 1) // 2
    if can_buy_units(mid):
        low = mid
    else:
        high = mid - 1

print(low)