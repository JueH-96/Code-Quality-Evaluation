def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    # D is a list of days for each month from month 1 to month N
    D = list(map(int, data[1:]))

    # A helper that returns True if all digits of number x are the same
    def is_repdigit(x):
        s = str(x)
        return all(ch == s[0] for ch in s)
        
    count = 0
    for month in range(1, N + 1):
        # Check if the month number is repdigit.
        if is_repdigit(month):
            digit = str(month)[0]
            
            # The only repdigit day numbers possible (within range up to 100)
            # are the one-digit: d, and the two-digit: dd (like 1 and 11)
            # (Three-digit repdigits start at 111, which is above any D_i <= 100)
            day1 = int(digit)   # single-digit repdigit date
            day2 = int(digit * 2)  # two-digit repdigit date
            
            # Use the current month's number of days D[month - 1]
            if day1 <= D[month - 1]:
                count += 1
            if day2 <= D[month - 1]:
                count += 1

    sys.stdout.write(str(count))
    
if __name__ == '__main__':
    main()