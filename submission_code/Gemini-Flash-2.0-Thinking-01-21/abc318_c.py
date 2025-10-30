import math

def solve():
    n, d, p = map(int, input().split())
    fares = list(map(int, input().split()))
    
    total_fare_sum = sum(fares)
    sorted_fares = sorted(fares, reverse=True)
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i+1] = prefix_sum[i] + sorted_fares[i]
        
    min_cost = float('inf')
    
    for m in range(n + 1):
        batches_needed = math.ceil(m / d) if d > 0 else 0
        pass_cost = batches_needed * p
        remaining_fare = total_fare_sum - prefix_sum[m]
        current_cost = pass_cost + remaining_fare
        min_cost = min(min_cost, current_cost)
        
    print(min_cost)

if __name__ == '__main__':
    solve()