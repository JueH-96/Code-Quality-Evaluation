def max_sleighs(reindeer_requirements, total_reindeer):
    sorted_requirements = sorted(reindeer_requirements)
    
    sleighs_pulled = 0
    total_reindeers_used = 0
    
    for req in sorted_requirements:
        if total_reindeers_used + req <= total_reindeer:
            total_reindeers_used += req
            sleighs_pulled += 1
        else:
            break
    
    return sleighs_pulled

def solve():
    N, Q = map(int, input().split())
    reindeer_requirements = list(map(int, input().split()))
    
    for _ in range(Q):
        X = int(input())
        result = max_sleighs(reindeer_requirements, X)
        print(result)

solve()