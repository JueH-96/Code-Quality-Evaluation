def solve():
    N = int(input())
    garbage_rules = []
    for _ in range(N):
        q, r = map(int, input().split())
        garbage_rules.append((q, r))
    
    Q = int(input())
    queries = []
    for _ in range(Q):
        t, d = map(int, input().split())
        queries.append((t, d))
    
    for t, d in queries:
        q, r = garbage_rules[t-1]
        
        next_collection_day = d
        while True:
            next_collection_day += 1
            if next_collection_day % q == r:
                print(next_collection_day)
                break

solve()