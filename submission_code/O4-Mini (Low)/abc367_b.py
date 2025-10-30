def main():
    s = input().strip()
    if '.' in s:
        integer_part, decimal_part = s.split('.')
        # Remove trailing zeros from decimal part
        decimal_part = decimal_part.rstrip('0')
        if decimal_part:
            print(f"{integer_part}.{decimal_part}")
        else:
            print(integer_part)
    else:
        # No decimal point (unlikely given spec), just print
        print(s)

if __name__ == "__main__":
    main()