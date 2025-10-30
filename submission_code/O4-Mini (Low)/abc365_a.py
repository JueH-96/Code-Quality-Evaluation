def main():
    Y = int(input().strip())
    # Leap year if divisible by 400, or divisible by 4 but not by 100
    if Y % 400 == 0 or (Y % 4 == 0 and Y % 100 != 0):
        print(366)
    else:
        print(365)

if __name__ == "__main__":
    main()