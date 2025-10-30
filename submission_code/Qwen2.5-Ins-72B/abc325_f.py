# YOUR CODE HERE
import sys
from functools import lru_cache

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = list(map(int, data[1:N+1]))
    L1, C1, K1 = map(int, data[N+1:N+4])
    L2, C2, K2 = map(int, data[N+4:N+7])
    
    @lru_cache(maxsize=None)
    def min_cost(d, k1, k2):
        if d <= 0:
            return 0
        if k1 == 0 and k2 == 0:
            return float('inf')
        
        cost1 = float('inf')
        if k1 > 0:
            cost1 = C1 + min_cost(d - L1, k1 - 1, k2)
        
        cost2 = float('inf')
        if k2 > 0:
            cost2 = C2 + min_cost(d - L2, k1, k2 - 1)
        
        return min(cost1, cost2)
    
    total_cost = 0
    for d in D:
        cost = min_cost(d, K1, K2)
        if cost == float('inf'):
            print(-1)
            return
        total_cost += cost
    
    print(total_cost)

solve()