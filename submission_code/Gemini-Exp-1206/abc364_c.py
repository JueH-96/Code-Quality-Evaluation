def solve():
    n, x, y = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    dishes = []
    for i in range(n):
        dishes.append((a[i], b[i]))
    
    ans = float('inf')
    
    import itertools
    
    for p in itertools.permutations(dishes):
        sweetness = 0
        saltiness = 0
        count = 0
        for dish in p:
            sweetness += dish[0]
            saltiness += dish[1]
            count += 1
            if sweetness > x or saltiness > y:
                ans = min(ans, count)
                break
        else:
            ans = min(ans, count)
            
    print(ans)

solve()