def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    M = int(data[0])
    days = list(map(int, data[1:]))
    
    total_days = sum(days)
    mid_day_index = (total_days + 1) // 2  # The middle day in 1-indexed sequence
    
    # Iterate through each month. Subtract the days until mid_day_index is within a month.
    for i in range(M):
        if mid_day_index > days[i]:
            mid_day_index -= days[i]
        else:
            # Found the month (i+1) and the day (mid_day_index)
            print(i + 1, mid_day_index)
            return

if __name__ == '__main__':
    main()