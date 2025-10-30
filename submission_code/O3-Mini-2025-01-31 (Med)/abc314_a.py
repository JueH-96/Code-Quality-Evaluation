def main():
    import sys
    input_line = sys.stdin.readline().strip()
    if not input_line:
        return
    N = int(input_line)
    # Pi constant to 100 decimal places as given in the problem
    pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"
    # Slice the string up to 2 + N characters (the "3." and N decimal digits)
    result = pi[:2+N]
    print(result)

if __name__ == '__main__':
    main()