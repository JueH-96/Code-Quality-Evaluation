s = input().strip()
integer_part, decimal_part = s.split('.')
processed_decimal = decimal_part.rstrip('0')
result = integer_part
if processed_decimal:
    result += '.' + processed_decimal
print(result)