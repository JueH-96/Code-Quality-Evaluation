s = input().strip()
if '.' in s:
    integer_part, fractional_part = s.split('.', 1)
    integer_clean = integer_part.lstrip('0')
    if integer_clean == '':
        integer_clean = '0'
    fractional_clean = fractional_part.rstrip('0')
    if fractional_clean == '':
        print(integer_clean)
    else:
        print(integer_clean + '.' + fractional_clean)
else:
    clean = s.lstrip('0')
    if clean == '':
        print('0')
    else:
        print(clean)