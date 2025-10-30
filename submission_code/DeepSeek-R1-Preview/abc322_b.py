n, m = map(int, input().split())
s = input().strip()
t = input().strip()

prefix = t.startswith(s)
suffix = t.endswith(s)

if prefix and suffix:
    print(0)
elif prefix:
    print(1)
elif suffix:
    print(2)
else:
    print(3)