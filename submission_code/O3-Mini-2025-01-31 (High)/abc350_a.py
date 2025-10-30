def main():
    import sys
    input_str = sys.stdin.readline().strip()
    # Extract the numeric part from the string, which is the last 3 characters.
    num = int(input_str[3:])
    
    # Check if the number is within the valid range [1, 349] and that it is not 316.
    if 1 <= num <= 349 and num != 316:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()