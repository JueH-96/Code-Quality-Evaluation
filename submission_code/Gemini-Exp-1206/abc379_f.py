def solve():
    n, q = map(int, input().split())
    h = list(map(int, input().split()))
    
    for _ in range(q):
        l, r = map(int, input().split())
        l -= 1
        r -= 1
        count = 0
        
        for j in range(r + 1, n):
            visible_from_l = True
            for k in range(l + 1, j):
                if k < r and h[k] > h[j]:
                    visible_from_l = False
                    break
                elif k > r and h[k] > h[j]:
                    visible_from_l = False
                    break
            
            visible_from_r = True
            for k in range(r + 1, j):
                if h[k] > h[j]:
                    visible_from_r = False
                    break
            
            if visible_from_l and visible_from_r:
                count += 1
        
        print(count)

solve()