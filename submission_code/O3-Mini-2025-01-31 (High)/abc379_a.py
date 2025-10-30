def main():
    import sys
    # Read the input as a string to easily extract each digit
    data = sys.stdin.read().strip().split()
    if data:
        s = data[0]
        # a, b, c correspond to the hundreds, tens, and ones digit respectively.
        a, b, c = s[0], s[1], s[2]
        # Form the numbers by rearranging the digits accordingly
        num1 = int(b + c + a)  # b, c, a order
        num2 = int(c + a + b)  # c, a, b order
        # Print the output as specified.
        sys.stdout.write(f"{num1} {num2}")

if __name__ == '__main__':
    main()