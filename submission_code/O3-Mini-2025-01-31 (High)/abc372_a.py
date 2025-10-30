def main():
    import sys
    # Read input string from standard input (strip to remove trailing newline)
    S = sys.stdin.readline().strip()
    # Remove all dots from S
    result = S.replace('.', '')
    # Print the result
    print(result)

# Call the main function
if __name__ == '__main__':
    main()