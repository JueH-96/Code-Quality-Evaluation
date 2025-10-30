x = input().strip()
if '.' in x:
    integer_part, decimal_part = x.split('.', 1)
    decimal_part = decimal_part.rstrip('0')
    if not decimal_part:
        if all(c == '0' for c in integer_part):
            print('0')
        else:
            print(integer_part)
    else:
        print(f"{integer_part}.{decimal_part}")
else:
    print(x)