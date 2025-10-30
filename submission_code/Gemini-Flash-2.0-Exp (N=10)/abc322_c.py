def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    results = []
    for i in range(1, n + 1):
        found = False
        for firework_day in a:
            if firework_day >= i:
                results.append(firework_day - i)
                found = True
                break
        
    for res in results:
        print(res)

solve()