x = input().strip()
integer_part, fractional_part = x.split('.')
fractional_part = fractional_part.rstrip('0')
if fractional_part:
    print(f"{integer_part}.{fractional_part}")
else:
    print(integer_part)