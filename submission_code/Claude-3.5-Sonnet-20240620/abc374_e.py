# YOUR CODE HERE
import sys

def max_production_capacity(N, X, machines):
    def can_produce(capacity):
        total_cost = 0
        for A, P, B, Q in machines:
            if A * P <= B * Q:
                units_needed = (capacity + A - 1) // A
                cost = units_needed * P
            else:
                units_needed = (capacity + B - 1) // B
                cost = units_needed * Q
            total_cost += cost
            if total_cost > X:
                return False
        return True

    left, right = 0, 10**10
    while right - left > 1:
        mid = (left + right) // 2
        if can_produce(mid):
            left = mid
        else:
            right = mid
    return left

# Read input
N, X = map(int, input().split())
machines = [tuple(map(int, input().split())) for _ in range(N)]

# Calculate and print the result
result = max_production_capacity(N, X, machines)
print(result)