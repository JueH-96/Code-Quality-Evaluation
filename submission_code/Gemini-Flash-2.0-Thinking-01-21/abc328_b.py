def is_repdigit(n):
    s = str(n)
    if not s:
        return False
    digit = s[0]
    for char in s:
        if char != digit:
            return False
    return True

def solve():
    n = int(input())
    d_values = list(map(int, input().split()))
    count = 0
    for month in range(1, n + 1):
        if is_repdigit(month):
            month_str = str(month)
            rep_digit = month_str[0]
            digit_val = int(rep_digit)
            days_in_month = d_values[month - 1]
            day_1digit = digit_val
            if day_1digit <= days_in_month:
                if is_repdigit(day_1digit):
                    count += 1
            day_2digit = digit_val * 10 + digit_val
            if day_2digit <= days_in_month:
                if is_repdigit(day_2digit):
                    count += 1
                    
    print(count)

if __name__ == '__main__':
    solve()