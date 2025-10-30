x_str = input()

if '.' not in x_str:
    print(x_str)
else:
    integer_part, decimal_part = x_str.split('.')
    processed_decimal_part = decimal_part.rstrip('0')
    if not processed_decimal_part:
        print(integer_part)
    else:
        print(f"{integer_part}.{processed_decimal_part}")