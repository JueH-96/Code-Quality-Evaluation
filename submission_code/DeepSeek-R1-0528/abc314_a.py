def main():
    decimal_part = "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    n = int(input().strip())
    if n == 0:
        print("3")
    else:
        print("3." + decimal_part[:n])

if __name__ == '__main__':
    main()