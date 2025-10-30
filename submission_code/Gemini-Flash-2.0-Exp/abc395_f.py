def solve():
    n, x = map(int, input().split())
    u = []
    d = []
    for _ in range(n):
        ui, di = map(int, input().split())
        u.append(ui)
        d.append(di)

    ans = float('inf')
    for h in range(1, 201):  # Iterate through possible values of H
        cost = 0
        valid = True
        u_new = []

        for i in range(n):
            if u[i] > h:
                valid = False
                break
            
            diff = u[i] + d[i] - h
            if diff > 0:
                cost += diff
                u_new.append(u[i] - diff)
            else:
                u_new.append(u[i])
        
        if not valid:
            continue
        
        
        valid_diff = True
        for i in range(n - 1):
            if abs(u_new[i] - u_new[i+1]) > x:
                valid_diff = False
                break
        
        if valid_diff:
            ans = min(ans, cost)

    
    min_h = min(u[i] + d[i] for i in range(n))
    max_h = max(u[i] + d[i] for i in range(n))
    
    for h in range(min_h, max_h + 1):
        cost = 0
        u_new = []
        
        for i in range(n):
            diff = u[i] + d[i] - h
            if diff > 0:
                cost += diff
                u_new.append(u[i] - diff)
            else:
                u_new.append(u[i])
        
        valid_diff = True
        for i in range(n - 1):
            if abs(u_new[i] - u_new[i+1]) > x:
                valid_diff = False
                break
        
        if valid_diff:
            ans = min(ans, cost)

    print(ans)

solve()