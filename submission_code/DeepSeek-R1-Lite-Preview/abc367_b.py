x = input().strip()
parts = x.split('.')
integer_part = parts[0]
decimal_part = parts[1].rstrip('0')
if decimal_part:
    print(f"{integer_part}.{decimal_part}")
else:
    print(integer_part)