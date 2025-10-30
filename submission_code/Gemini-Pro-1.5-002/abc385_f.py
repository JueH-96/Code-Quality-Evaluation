def solve():
    n = int(input())
    buildings = []
    for _ in range(n):
        x, h = map(int, input().split())
        buildings.append((x, h))

    def can_see_all(height):
        for i in range(n):
            visible = False
            for j in range(n):
                if i == j:
                    continue
                
                intersect = False
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    x1, h1 = 0, height
                    x2, h2 = buildings[i][0], buildings[i][1]
                    x3, h3 = buildings[k][0], buildings[k][1]
                    
                    if (x2 - x1) * (h3 - h1) - (x3 - x1) * (h2 - h1) > 0 and (x2-x1)*(buildings[j][1]-h1) - (buildings[j][0]-x1)*(h2-h1) < 0:
                        intersect = True
                        break
                        
                if not intersect:
                    x1, h1 = 0, height
                    x2, h2 = buildings[i][0], buildings[i][1]
                    x3, h3 = buildings[j][0], buildings[j][1]
                    if (x2 - x1) * (h3 - h1) - (x3 - x1) * (h2 - h1) <= 0:
                        visible = True
                        break
            if not visible:
                return False
        return True

    if can_see_all(0):
        print("-1")
        return

    low = 0
    high = 10**10
    ans = 0
    for _ in range(100):
        mid = (low + high) / 2
        if not can_see_all(mid):
            ans = mid
            high = mid
        else:
            low = mid
    
    print(ans)

solve()