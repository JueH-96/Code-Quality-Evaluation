def solve():
    n = int(input())
    bases = []
    for _ in range(n):
        w, x = map(int, input().split())
        bases.append((w, x))

    max_employees = 0
    for start_time in range(24):
        end_time = (start_time + 1) % 24
        
        current_employees = 0
        for w, x in bases:
            start_base_time = (start_time + x) % 24
            end_base_time = (end_time + x) % 24
            
            if start_base_time < end_base_time:
                if 9 <= start_base_time and end_base_time <= 18:
                    current_employees += w
            else:
                if 9 <= start_base_time or end_base_time <= 18:
                    if (9 <= start_base_time and start_base_time <= 17) or (10 <= end_base_time and end_base_time <= 18) or (9 <= start_base_time and end_base_time <= 18):
                        if (9 <= start_base_time and start_base_time <= 17) and (10 <= end_base_time and end_base_time <= 18):
                            current_employees += w
                        elif (9 <= start_base_time and start_base_time <= 17) and (end_base_time < 9 or end_base_time > 18):
                            pass
                        elif (start_base_time < 9 or start_base_time > 17) and (10 <= end_base_time and end_base_time <= 18):
                            pass
                        else:
                            pass
                    else:
                        pass
        max_employees = max(max_employees, current_employees)

    print(max_employees)

solve()