def solve():
    n, d = map(int, input().split())
    schedules = [input() for _ in range(n)]
    
    max_consecutive_days = 0
    
    for i in range(d):
        for j in range(i, d):
            consecutive_days = j - i + 1
            
            all_free = True
            for day in range(i, j + 1):
                for person_schedule in schedules:
                    if person_schedule[day] == 'x':
                        all_free = False
                        break
                if not all_free:
                    break
            
            if all_free:
                max_consecutive_days = max(max_consecutive_days, consecutive_days)
    
    print(max_consecutive_days)

solve()