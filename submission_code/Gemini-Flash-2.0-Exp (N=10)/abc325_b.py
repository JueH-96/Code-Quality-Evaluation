def solve():
    n = int(input())
    bases = []
    for _ in range(n):
        w, x = map(int, input().split())
        bases.append((w, x))

    max_employees = 0
    for start_time_utc in range(24):
        end_time_utc = start_time_utc + 1
        current_employees = 0
        for w, x in bases:
            start_time_local = (start_time_utc + x) % 24
            end_time_local = (end_time_utc + x) % 24
            
            if start_time_local < end_time_local:
                if 9 <= start_time_local and end_time_local <= 18:
                    current_employees += w
            else:
                if (9 <= start_time_local or end_time_local <= 18) and (start_time_local < 18 or end_time_local > 9):
                    if (9 <= start_time_local and start_time_local < 18) or (9 < end_time_local and end_time_local <= 18):
                        if (9 <= start_time_local and end_time_local <= 18) or (start_time_local < 18 and end_time_local > 9):
                            current_employees += w
        max_employees = max(max_employees, current_employees)
    print(max_employees)

solve()