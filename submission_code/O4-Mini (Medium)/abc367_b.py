def main():
    s = input().strip()
    # If there's no decimal point, just print as is
    if '.' not in s:
        print(s)
        return

    integer_part, decimal_part = s.split('.')
    # Remove trailing zeros from the decimal part
    decimal_part = decimal_part.rstrip('0')
    # If the decimal part is now empty, just print the integer part
    if decimal_part == '':
        print(integer_part)
    else:
        print(f"{integer_part}.{decimal_part}")

if __name__ == "__main__":
    main()