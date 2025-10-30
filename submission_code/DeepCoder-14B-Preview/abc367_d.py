x = input().strip()
parts = x.split('.', 1)

if len(parts) == 1:
    integer_part = parts[0]
    fractional_part = '000'
else:
    integer_part = parts[0]
    fractional_part = parts[1]

stripped = fractional_part.rstrip('0')

if stripped:
    print(f"{integer_part}.{stripped}")
else:
    print(integer_part)