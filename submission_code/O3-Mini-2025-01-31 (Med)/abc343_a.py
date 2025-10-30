def main():
    import sys
    # Read inputs from standard input. Expecting two integers.
    data = sys.stdin.read().split()
    if not data:
        return
    A, B = map(int, data[:2])
    total = A + B
    # Choose an integer between 0 and 9 that is not equal to A+B.
    # We'll choose 0 if it is not equal, else choose 1.
    if total != 0:
        result = 0
    else:
        result = 1
    # Print the resultant integer.
    print(result)

# Call main to execute the solution
if __name__ == '__main__':
    main()