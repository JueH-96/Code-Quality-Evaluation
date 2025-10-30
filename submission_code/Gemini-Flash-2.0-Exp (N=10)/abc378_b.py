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
        queries.append((t - 1, d))
    
    for t, d in queries:
        q, r = garbage_rules[t]
        
        if d % q == r:
            print(d)
            continue
        
        next_collection_day = ((d // q) + 1) * q + r
        if next_collection_day <= d:
            next_collection_day = ((d // q) + 1) * q + r
        
        if d % q > r:
            next_collection_day = ((d // q) + 1) * q + r
        elif d % q < r:
            next_collection_day = ((d // q) ) * q + r
            if next_collection_day <= d:
                next_collection_day = ((d // q) + 1) * q + r
        
        print(next_collection_day)

solve()