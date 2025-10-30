import math

def solve():
    n, d, p = map(int, input().split())
    fares = list(map(int, input().split()))
    sorted_fares = sorted(fares, reverse=True)
    prefix_sum_fares = [0] * (n + 1)
    for i in range(n):
        prefix_sum_fares[i+1] = prefix_sum_fares[i] + sorted_fares[i]
    total_fare_sum = prefix_sum_fares[n]
    min_cost = float('inf')
    for k in range(n + 1):
        num_batches = math.ceil(k / d) if k > 0 else 0
        pass_cost = num_batches * p
        fare_cost = total_fare_sum - prefix_sum_fares[k]
        current_cost = pass_cost + fare_cost
        min_cost = min(min_cost, current_cost)
    print(min_cost)

if __name__ == '__main__':
    solve()