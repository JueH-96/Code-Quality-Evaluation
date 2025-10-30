# YOUR CODE HERE
X = input().strip()

# Remove trailing zeros after the decimal point
if '.' in X:
    # Split into integer and decimal parts
    parts = X.split('.')
    integer_part = parts[0]
    decimal_part = parts[1]
    
    # Remove trailing zeros from decimal part
    decimal_part = decimal_part.rstrip('0')
    
    # If decimal part is empty, just print integer part
    if decimal_part == '':
        print(integer_part)
    else:
        print(integer_part + '.' + decimal_part)
else:
    # No decimal point, just print as is
    print(X)