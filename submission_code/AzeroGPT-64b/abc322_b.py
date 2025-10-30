n, m = map(int, input().split())
s = input()
t = input()

is_prefix = (t.startswith(s))
is_suffix = (t.endswith(s))

if is_prefix and is_suffix:
    ans = 0
elif is_prefix:
    ans = 1
elif is_suffix:
    ans = 2
else:
    ans = 3 
print(ans)