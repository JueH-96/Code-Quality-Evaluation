x_str = input()
parts = x_str.split('.')
integer_part = parts[0]
decimal_part = parts[1]

trimmed_decimal_part = decimal_part.rstrip('0')

if not trimmed_decimal_part:
    print(integer_part)
else:
    print(integer_part + '.' + trimmed_decimal_part)