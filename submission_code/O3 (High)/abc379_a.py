def main():
    import sys

    # Read the three-digit number as a string to easily access each digit
    s = sys.stdin.readline().strip()

    # Extract the digits
    a, b, c = s[0], s[1], s[2]   # a: hundreds, b: tens, c: ones

    # Form the required numbers
    bca = int(b + c + a)
    cab = int(c + a + b)

    # Output
    print(f"{bca} {cab}")

if __name__ == "__main__":
    main()