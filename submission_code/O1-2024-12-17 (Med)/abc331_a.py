def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    M, D = map(int, data[:2])  # M months per year, D days per month
    y, m, d = map(int, data[2:])  # current date
    
    # Compute the next day
    if d < D:
        # If not the last day of the month, simply increment the day
        d += 1
    else:
        # If it is the last day of the month, reset to day 1
        d = 1
        if m < M:
            # If not the last month, move to the next month
            m += 1
        else:
            # Else move to the next year and reset month to 1
            m = 1
            y += 1
    
    print(y, m, d)

# Do not forget to call main here
main()