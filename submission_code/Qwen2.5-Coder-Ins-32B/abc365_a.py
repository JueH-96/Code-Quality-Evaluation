# YOUR CODE HERE
import sys

def is_leap_year(year):
    if year % 4 != 0:
        return False
    elif year % 100 != 0:
        return True
    elif year % 400 != 0:
        return False
    else:
        return True

def main():
    input = sys.stdin.read().strip()
    year = int(input)
    if is_leap_year(year):
        print(366)
    else:
        print(365)

if __name__ == "__main__":
    main()