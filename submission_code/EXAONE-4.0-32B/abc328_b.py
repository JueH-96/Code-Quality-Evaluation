def main():
    n = int(input().strip())
    D_list = list(map(int, input().split()))
    
    count = 0
    for month in range(1, n + 1):
        days_in_month = D_list[month - 1]
        for day in range(1, days_in_month + 1):
            s = str(month) + str(day)
            if len(set(s)) == 1:
                count += 1
                
    print(count)

if __name__ == "__main__":
    main()