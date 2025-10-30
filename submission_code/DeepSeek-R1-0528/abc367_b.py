s = input().strip()

if '.' in s:
    parts = s.split('.', 1)
    a_str = parts[0]
    b_str = parts[1]
    if a_str == '':
        a_clean = '0'
    else:
        a_clean = str(int(a_str))
    b_clean = b_str.rstrip('0')
    if b_clean == '':
        print(a_clean)
    else:
        print(a_clean + '.' + b_clean)
else:
    if s == '':
        print('0')
    else:
        num = int(s)
        print(str(num))