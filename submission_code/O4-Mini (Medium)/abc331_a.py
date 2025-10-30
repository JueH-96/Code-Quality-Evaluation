def main():
    # Read the number of months M and days per month D
    M, D = map(int, input().split())
    # Read the current date y (year), m (month), d (day)
    y, m, d = map(int, input().split())

    # Advance one day
    d += 1
    # If we exceed the days in a month, roll over to next month
    if d > D:
        d = 1
        m += 1
        # If we exceed the months in a year, roll over to next year
        if m > M:
            m = 1
            y += 1

    # Print the updated date
    print(y, m, d)

if __name__ == "__main__":
    main()