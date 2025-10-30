def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, D = map(int, data[:2])
    schedules = data[2:]

    # Initialize an array to track if all people are free on each day
    free_days = [True] * D

    # Determine which days are free for all people
    for i in range(N):
        for j in range(D):
            if schedules[i][j] == 'x':
                free_days[j] = False

    # Find the maximum consecutive stretch of True in free_days
    max_streak = 0
    current_streak = 0
    for j in range(D):
        if free_days[j]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    print(max_streak)

# Do not forget to call main()!
if __name__ == "__main__":
    main()