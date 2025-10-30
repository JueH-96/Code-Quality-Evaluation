def is_repdigit(n):
    s = str(n)
    if not s:
        return False, ''
    digit = s[0]
    for char in s:
        if char != digit:
            return False, ''
    return True, digit

def solve():
    n_months = int(input())
    days_in_month = list(map(int, input().split()))
    
    total_repdigit_dates = 0
    for month_index in range(n_months):
        month_number = month_index + 1
        is_month_rep, month_digit = is_repdigit(month_number)
        if is_month_rep:
            current_day_value = int(month_digit)
            max_days_in_month = days_in_month[month_index]
            while current_day_value <= max_days_in_month:
                is_day_rep, day_digit = is_repdigit(current_day_value)
                if is_day_rep and day_digit == month_digit:
                    total_repdigit_dates += 1
                current_day_value = current_day_value * 10 + int(month_digit)
                
    print(total_repdigit_dates)

if __name__ == '__main__':
    solve()