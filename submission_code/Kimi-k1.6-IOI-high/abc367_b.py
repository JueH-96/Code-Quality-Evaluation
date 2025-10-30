x = input().strip()
if '.' in x:
    a, b = x.split('.')
    b = b.rstrip('0')
    print(f"{a}{f'.{b}' if b else ''}")
else:
    print(x)