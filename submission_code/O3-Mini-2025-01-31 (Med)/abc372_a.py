def main():
    import sys
    input_str = sys.stdin.read().strip()
    # Remove all periods from the string
    result = input_str.replace('.', '')
    # Print the result
    print(result)

if __name__ == '__main__':
    main()