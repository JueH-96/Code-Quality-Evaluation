def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    M, D = map(int, data[:2])
    y, m, d = map(int, data[2:])
    
    # Check if it's the last day of the month
    if d < D:
        # Move to the next day in the same month
        d += 1
    else:
        # If it's the last day of the month, reset the day and check if it's the last month of the year
        d = 1
        if m < M:
            m += 1
        else:
            m = 1
            y += 1
    
    print(y, m, d)

# Do not forget to call main().
main()