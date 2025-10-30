s = input().strip()
if '.' in s:
    int_part, frac_part = s.split('.', 1)
    processed_frac = frac_part[::-1].lstrip('0')[::-1]
    if processed_frac:
        print(f"{int_part}.{processed_frac}")
    else:
        print(int_part)
else:
    print(s)