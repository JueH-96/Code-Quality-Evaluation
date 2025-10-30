def max_units(N, M, P):
    def can_buy(total_units):
        remaining_budget = M
        for p in P:
            # Calculate maximum units we can buy for this product
            # Solve k^2 * p <= remaining_budget
            if remaining_budget <= 0:
                return False
            max_k = int((remaining_budget // p) ** 0.5)
            if total_units <= max_k:
                remaining_budget -= total_units * total_units * p
                return remaining_budget >= 0
            else:
                remaining_budget -= max_k * max_k * p
                total_units -= max_k
        return total_units <= 0

    left, right = 0, 10**18
    while left < right:
        mid = (left + right + 1) // 2
        if can_buy(mid):
            left = mid
        else:
            right = mid - 1
    return left

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
P = list(map(int, data[2:]))

print(max_units(N, M, P))