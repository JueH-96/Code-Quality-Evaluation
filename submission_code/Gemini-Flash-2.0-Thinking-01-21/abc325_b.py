def solve():
    n = int(input())
    bases = []
    for _ in range(n):
        w_x = list(map(int, input().split()))
        bases.append({'employees': w_x[0], 'x_shift': w_x[1]})
    
    max_employees = 0
    for start_hour_utc in range(24):
        current_employees = 0
        for base in bases:
            w = base['employees']
            x_shift = base['x_shift']
            local_start_hour = (start_hour_utc + x_shift) % 24
            local_end_hour = (start_hour_utc + 1 + x_shift) % 24
            
            valid_meeting = False
            if 9 <= local_start_hour <= 18 and 9 <= local_end_hour <= 18:
                valid_meeting = True
                
            if valid_meeting:
                current_employees += w
                
        max_employees = max(max_employees, current_employees)
        
    print(max_employees)

if __name__ == '__main__':
    solve()