def is_same_digit(m, d):
    """
    Check if all digits in the decimal notations of month m and day d are the same.
    """
    month_str = str(m)
    day_str = str(d)
    
    all_digits = month_str + day_str
    return len(set(all_digits)) == 1

def main():
    N = int(input())
    days_per_month = list(map(int, input().split()))
    
    repdigit_dates_count = 0
    
    for month in range(1, N + 1):
        for day in range(1, days_per_month[month - 1] + 1):
            if is_same_digit(month, day):
                repdigit_dates_count += 1
    
    print(repdigit_dates_count)

if __name__ == "__main__":
    main()