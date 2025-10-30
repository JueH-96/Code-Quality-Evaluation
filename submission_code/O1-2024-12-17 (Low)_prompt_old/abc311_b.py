def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, D = map(int, data[:2])        # Read N (number of people) and D (number of days)
    schedules = data[2:]            # Next N lines (each string of length D)

    # Initialize a list indicating if all people are free on a given day
    # Start by assuming every day is free for everyone.
    free_days = [True]*D

    # For each person, for each day, set free_days[j] = False if that person is busy on day j
    for i in range(N):
        for j, c in enumerate(schedules[i]):
            if c == 'x':
                free_days[j] = False

    # Find the maximum length of consecutive True in free_days
    max_streak = 0
    current_streak = 0

    for i in range(D):
        if free_days[i]:
            current_streak += 1
        else:
            max_streak = max(max_streak, current_streak)
            current_streak = 0

    # After the loop, compare again in case the last streak goes till the end
    max_streak = max(max_streak, current_streak)

    # Print the maximum streak
    print(max_streak)

def main():
    solve()

if __name__ == "__main__":
    main()