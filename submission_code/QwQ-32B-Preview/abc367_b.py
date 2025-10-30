def main():
    # Read input as string
    X = input().strip()
    
    # Split into integer and decimal parts
    integer_part, decimal_part = X.split('.')
    
    # Remove trailing zeros from decimal part
    decimal_part_stripped = decimal_part.rstrip('0')
    
    # Construct and print the result
    if decimal_part_stripped:
        print(f"{integer_part}.{decimal_part_stripped}")
    else:
        print(integer_part)

if __name__ == "__main__":
    main()