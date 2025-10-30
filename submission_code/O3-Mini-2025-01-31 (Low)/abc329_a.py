def main():
    import sys
    input_str = sys.stdin.read().strip()  # Read input from standard input, remove trailing newline
    # The problem statement says the input is just one string, so input_str itself is S.
    # Separate each character with a space and print.
    print(" ".join(input_str))

if __name__ == '__main__':
    main()