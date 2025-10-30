def main():
    import sys

    data = sys.stdin.read().strip().split()
    M = int(data[0])
    D = list(map(int, data[1:]))

    # The total number of days in the year (which is odd)
    total_days = sum(D)
    # The middle day index (1-based)
    mid_day = (total_days + 1) // 2

    running_sum = 0
    for i in range(M):
        if running_sum + D[i] >= mid_day:
            day_in_month = mid_day - running_sum
            print(i + 1, day_in_month)
            return
        running_sum += D[i]

# Do not forget to call main()
main()