import sys

def can_schedule_all_plans_on_holidays(N, A, B, D):
    # Calculate the total number of days in a week
    total_days = A + B

    # Create a set to store the holiday days in the first week
    holidays = set(range(1, A + 1))

    # Check each plan day
    for d in D:
        # Calculate the day of the week for the plan day
        day_of_week = (d % total_days)
        if day_of_week == 0:
            day_of_week = total_days

        # Check if the day of the week is a holiday
        if day_of_week not in holidays:
            return "No"

    return "Yes"

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    D = list(map(int, data[3:]))

    result = can_schedule_all_plans_on_holidays(N, A, B, D)
    print(result)

if __name__ == "__main__":
    main()