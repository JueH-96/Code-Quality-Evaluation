x_str = input()
if '.' not in x_str:
    print(x_str)
else:
    integer_part, decimal_part = x_str.split('.')
    decimal_part = decimal_part.rstrip('0')
    if not decimal_part:
        print(integer_part)
    else:
        print(f"{integer_part}.{decimal_part}")