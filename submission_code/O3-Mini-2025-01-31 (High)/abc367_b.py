def main():
    import sys
    # Read input and remove any extra whitespace including newlines
    s = sys.stdin.read().strip()
    
    # Check if the string contains a decimal point
    if '.' in s:
        # Remove trailing zeros from the fractional part
        s = s.rstrip('0')
        # If there is a dangling decimal point, remove it
        if s.endswith('.'):
            s = s[:-1]
    print(s)

# Call the main function
if __name__ == '__main__':
    main()