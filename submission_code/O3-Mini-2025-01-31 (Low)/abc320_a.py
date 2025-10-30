def main():
    import sys
    # Read input from standard input
    input_line = sys.stdin.read().strip()
    if not input_line:
        return
    # Split the input by whitespace and convert to integers
    A, B = map(int, input_line.split())
    # Calculate the result: A^B + B^A using Python exponentiation
    result = (A ** B) + (B ** A)
    # Print the result
    print(result)

if __name__ == '__main__':
    main()