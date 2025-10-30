def solve():
    n = int(input())
    ranges = []
    for _ in range(n):
        l, r = map(int, input().split())
        ranges.append((l, r))
    
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(int(input()))
    
    for x in queries:
        rating = x
        for l, r in ranges:
            if l <= rating <= r:
                rating += 1
        print(rating)

solve()