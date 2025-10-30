def solve():
    n = int(input())
    d = list(map(int, input().split()))
    
    count = 0
    for i in range(1, n + 1):
        for j in range(1, d[i-1] + 1):
            month_str = str(i)
            day_str = str(j)
            
            if len(set(month_str)) == 1 and len(set(day_str)) == 1 and month_str[0] == day_str[0]:
                count += 1
                
    print(count)

solve()