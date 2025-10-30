def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    days = list(map(int, data[1:]))
    
    count = 0
    # Iterate over each month i from 1 to N.
    for i in range(1, N + 1):
        # Check if the month number i is a repdigit (i.e. all digits are the same)
        s_month = str(i)
        if len(set(s_month)) != 1:
            continue  # if month i isn't repdigit, then no date in that month can be
        
        # For this month, we only want day j that are repdigit with the same repeating digit.
        rep_digit = s_month[0]  # all digits in month i are rep_digit
        max_day = days[i - 1]
        for j in range(1, max_day + 1):
            s_day = str(j)
            # Check if all digits in day j are the same AND they match the month digit.
            if len(set(s_day)) == 1 and s_day[0] == rep_digit:
                count += 1
                
    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()