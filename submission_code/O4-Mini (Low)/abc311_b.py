def main():
    import sys
    input = sys.stdin.readline

    # Read N (number of people) and D (number of days)
    N, D = map(int, input().split())

    # Read schedules
    schedules = [input().strip() for _ in range(N)]

    # For each day, determine if all people are free ('o')
    free_days = [all(schedules[i][d] == 'o' for i in range(N)) for d in range(D)]

    # Find the maximum length of consecutive True in free_days
    max_consec = 0
    current = 0
    for is_free in free_days:
        if is_free:
            current += 1
            if current > max_consec:
                max_consec = current
        else:
            current = 0

    print(max_consec)


if __name__ == "__main__":
    main()