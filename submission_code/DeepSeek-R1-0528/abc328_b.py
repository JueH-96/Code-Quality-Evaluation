def main():
    N = int(input().strip())
    D_list = list(map(int, input().split()))
    
    count = 0
    for d in range(1, 10):
        months = []
        if d <= N:
            months.append(d)
        if 11 * d <= N:
            months.append(11 * d)
            
        for m in months:
            days_in_month = D_list[m - 1]
            day1 = d
            day2 = 11 * d
            
            if day1 <= days_in_month:
                count += 1
            if day2 <= days_in_month:
                count += 1
                
    print(count)

if __name__ == '__main__':
    main()