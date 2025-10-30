def main():
    pi_decimals = "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    N = int(input().strip())
    # Print "3." followed by the first N digits of pi_decimal
    print("3." + pi_decimals[:N])

if __name__ == "__main__":
    main()