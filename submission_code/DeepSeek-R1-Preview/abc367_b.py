s = input().strip()
parts = s.split('.')
integer_part = parts[0]
fractional_part = parts[1]

# Process integer part to remove leading zeros
integer_processed = str(int(integer_part))

# Process fractional part to remove trailing zeros
fractional_stripped = fractional_part.rstrip('0')

if len(fractional_stripped) == 0:
    print(integer_processed)
else:
    print(f"{integer_processed}.{fractional_stripped}")