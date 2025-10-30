import sys
from itertools import combinations

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = int(data[2])
    idx = 3
    plans = []
    for _ in range(N):
        C = int(data[idx])
        A = list(map(int, data[idx+1:idx+K+1]))
        plans.append((C, A))
        idx += K + 1
    
    min_cost = float('inf')
    for r in range(1, N+1):
        for comb in combinations(plans, r):
            total_cost = sum(c for c, _ in comb)
            params = [0] * K
            for _, A in comb:
                for j in range(K):
                    params[j] += A[j]
            if all(p >= P for p in params):
                min_cost = min(min_cost, total_cost)
    
    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)

solve()