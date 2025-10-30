import sys

def is_leap_year(year: int) -> bool:
    """
    Returns True if 'year' is a leap year in the Gregorian calendar, False otherwise.
    """
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    return year % 4 == 0

def main() -> None:
    y_str = sys.stdin.readline().strip()
    if not y_str:
        return
    year = int(y_str)
    days = 366 if is_leap_year(year) else 365
    print(days)

if __name__ == "__main__":
    main()