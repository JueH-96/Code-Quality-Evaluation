x = input().strip()
integer_part, decimal_part = x.split('.')
integer_part = str(int(integer_part))
trimmed_decimal = decimal_part.rstrip('0')
if trimmed_decimal:
    print(f"{integer_part}.{trimmed_decimal}")
else:
    print(integer_part)