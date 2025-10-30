def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    N = int(input_data)
    
    s = str(N)
    length = len(s)
    
    # If N has 3 or fewer digits, print as is.
    if length <= 3:
        print(N)
    else:
        # For numbers with 4 to 9 digits, we truncate the last (length - 3) digits.
        factor = 10 ** (length - 3)
        result = (N // factor) * factor
        print(result)

if __name__ == '__main__':
    main()