def solve():
    n = int(input())
    bases = []
    for _ in range(n):
        w, x = map(int, input().split())
        bases.append({'employees': w, 'offset': x})
    
    employee_counts = [0] * 24
    
    for base in bases:
        employees = base['employees']
        offset = base['offset']
        for j in range(9, 18):
            start_utc_time = (j - offset) % 24
            employee_counts[start_utc_time] += employees
            
    max_employees = 0
    for count in employee_counts:
        max_employees = max(max_employees, count)
        
    print(max_employees)

if __name__ == '__main__':
    solve()