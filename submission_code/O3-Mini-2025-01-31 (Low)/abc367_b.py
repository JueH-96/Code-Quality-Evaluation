def main():
    import sys
    input_str = sys.stdin.read().strip()
    if not input_str:
        return

    # Remove any trailing zeros after the decimal point.
    if '.' in input_str:
        # Remove trailing zeros.
        input_str = input_str.rstrip('0')
        # If the string ends with '.', it means the entire decimal portion was removed; so remove the decimal point.
        if input_str[-1] == '.':
            input_str = input_str[:-1]
    
    sys.stdout.write(input_str)

if __name__ == '__main__':
    main()