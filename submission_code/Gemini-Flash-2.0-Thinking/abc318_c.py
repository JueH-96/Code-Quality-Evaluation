import math

def solve():
    n, d, p = map(int, input().split())
    f = list(map(int, input().split()))
    f.sort()

    min_cost = float('inf')

    for r in range(n + 1):
        regular_fare_days = r
        days_covered_by_passes = n - r

        if days_covered_by_passes > 0:
            num_batches = (days_covered_by_passes + d - 1) // d
            cost_of_passes = num_batches * p
        else:
            cost_of_passes = 0

        cost_of_regular_fares = sum(f[:regular_fare_days])

        total_cost = cost_of_passes + cost_of_regular_fares
        min_cost = min(min_cost, total_cost)

    print(min_cost)

solve()