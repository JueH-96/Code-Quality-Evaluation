def solve():
    n = int(input())
    d = list(map(int, input().split()))
    
    count = 0
    for i in range(n):
        month = i + 1
        for day in range(1, d[i] + 1):
            month_str = str(month)
            day_str = str(day)
            
            if len(month_str) == 1 and len(day_str) == 1:
                if month_str == day_str:
                    count += 1
            elif len(month_str) == 1 and len(day_str) == 2:
                if day_str[0] == day_str[1] and month_str == day_str[0]:
                    count += 1
            elif len(month_str) == 2 and len(day_str) == 1:
                if month_str[0] == month_str[1] and month_str[0] == day_str:
                    count += 1
            elif len(month_str) == 2 and len(day_str) == 2:
                if month_str[0] == month_str[1] and day_str[0] == day_str[1] and month_str[0] == day_str[0]:
                    count += 1
    print(count)

solve()