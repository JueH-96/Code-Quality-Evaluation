from itertools import combinations

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = int(data[2])
    
    plans = []
    index = 3
    for _ in range(N):
        cost = int(data[index])
        increases = list(map(int, data[index+1:index+1+K]))
        index += 1 + K
        plans.append((cost, increases))
    
    # We need to achieve at least P in each of the K parameters
    min_cost = float('inf')
    found = False
    
    # Check all possible combinations of plans
    for r in range(1, N+1):
        for combo in combinations(plans, r):
            total_cost = sum(plan[0] for plan in combo)
            total_increases = [0] * K
            for plan in combo:
                for j in range(K):
                    total_increases[j] += plan[1][j]
            if all(x >= P for x in total_increases):
                found = True
                min_cost = min(min_cost, total_cost)
    
    if found:
        print(min_cost)
    else:
        print(-1)