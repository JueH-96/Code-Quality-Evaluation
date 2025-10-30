def solve():
    n, m = map(int, input().split())
    noodles = [0] * n
    returns = []
    
    for _ in range(m):
        t, w, s = map(int, input().split())
        
        available = []
        for i in range(n):
            available.append(i + 1)
        
        
        row = []
        
        for i in range(n):
            
            present = True
            
            for return_time, person_id in returns:
                if person_id == i+1 and return_time > t:
                    present = False
                    break
            
            if present:
                row.append(i+1)
        
        if len(row) > 0:
            person_at_front = row[0]
            noodles[person_at_front - 1] += w
            returns.append((t + s, person_at_front))
            
    for amount in noodles:
        print(amount)

solve()