# YOUR CODE HERE
def count_repdigit_days():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    days_in_months = list(map(int, data[1:]))
    
    repdigit_days_count = 0
    
    for month in range(1, N + 1):
        for day in range(1, days_in_months[month - 1] + 1):
            month_str = str(month)
            day_str = str(day)
            
            # Check if month_str is a repdigit
            if len(set(month_str)) == 1:
                # Check if day_str is a repdigit
                if len(set(day_str)) == 1:
                    repdigit_days_count += 1
    
    print(repdigit_days_count)

count_repdigit_days()