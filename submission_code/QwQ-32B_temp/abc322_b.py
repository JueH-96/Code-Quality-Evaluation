n, m = map(int, input().split())
s = input().strip()
t = input().strip()

is_prefix = s == t[:n]
is_suffix = s == t[-n:]

if is_prefix and is_suffix:
    print(0)
elif is_prefix:
    print(1)
elif is_suffix:
    print(2)
else:
    print(3)