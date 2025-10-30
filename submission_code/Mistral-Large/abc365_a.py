import sys

def is_leap_year(year):
    if year % 400 == 0:
        return 366
    elif year % 100 == 0:
        return 365
    elif year % 4 == 0:
        return 366
    else:
        return 365

def main():
    input = sys.stdin.read()
    year = int(input.strip())
    print(is_leap_year(year))

if __name__ == "__main__":
    main()