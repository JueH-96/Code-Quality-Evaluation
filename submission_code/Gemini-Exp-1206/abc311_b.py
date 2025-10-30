def solve():
    n, d = map(int, input().split())
    schedules = [input() for _ in range(n)]
    
    max_consecutive_days = 0
    for i in range(d):
        for j in range(i, d):
            all_free = True
            for k in range(n):
                for day in range(i, j + 1):
                    if schedules[k][day] == 'x':
                        all_free = False
                        break
                if not all_free:
                    break
            
            if all_free:
                max_consecutive_days = max(max_consecutive_days, j - i + 1)
    
    print(max_consecutive_days)

solve()