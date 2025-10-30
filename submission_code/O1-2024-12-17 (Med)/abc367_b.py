def main():
    s = input().strip()
    # Split into integer and decimal parts
    integer_part, decimal_part = s.split('.')
    
    # Strip trailing zeroes from the decimal part
    decimal_part = decimal_part.rstrip('0')
    
    # If decimal part becomes empty after stripping, just print the integer part
    if not decimal_part:
        print(integer_part)
    else:
        print(f"{integer_part}.{decimal_part}")

# Don't forget to call main!
main()