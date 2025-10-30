def main():
    import sys
    data = sys.stdin.read().split()
    M = int(data[0])
    D = int(data[1])
    y = int(data[2])
    m = int(data[3])
    d = int(data[4])
    
    # If it's not the last day of the month, just increment the day.
    if d < D:
        d += 1
    else:
        # It's the last day of the month, so reset day and maybe increment month.
        d = 1
        if m < M:
            m += 1
        else:
            # It's the last month of the year, so reset month and increment year.
            m = 1
            y += 1
    print(y, m, d)

if __name__ == '__main__':
    main()