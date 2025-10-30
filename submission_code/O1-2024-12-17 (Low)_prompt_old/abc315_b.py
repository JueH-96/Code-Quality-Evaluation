def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    M = int(data[0])
    D = list(map(int, data[1:]))

    total_days = sum(D)
    middle_day = (total_days + 1) // 2  # integer index of the middle day

    running_sum = 0
    for month_idx in range(M):
        if running_sum + D[month_idx] >= middle_day:
            day_in_month = middle_day - running_sum
            print(month_idx + 1, day_in_month)
            return
        running_sum += D[month_idx]

def main():
    solve()

if __name__ == "__main__":
    main()