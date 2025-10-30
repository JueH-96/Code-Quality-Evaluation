x = input().strip()
a, b = x.split('.')
b_stripped = b.rstrip('0')
if len(b_stripped) == 0:
    print(a)
else:
    print(f"{a}.{b_stripped}")