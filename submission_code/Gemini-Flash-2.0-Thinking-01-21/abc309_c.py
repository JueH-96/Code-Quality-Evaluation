def solve():
    n, k = map(int, input().split())
    medicines = []
    for _ in range(n):
        medicines.append(list(map(int, input().split())))
    
    days_to_check = set()
    days_to_check.add(1)
    for a_i, b_i in medicines:
        days_to_check.add(a_i + 1)
        
    first_day = float('inf')
    
    sorted_days = sorted(list(days_to_check))
    
    for day in sorted_days:
        if day <= 0:
            continue
        total_pills = 0
        for a_i, b_i in medicines:
            if day <= a_i:
                total_pills += b_i
                
        if total_pills <= k:
            first_day = min(first_day, day)
            break
            
    print(first_day)

if __name__ == '__main__':
    solve()