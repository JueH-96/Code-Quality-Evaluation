N, M = map(int, input().split())
S = input()
T = input()

is_prefix = T.startswith(S)
is_suffix = T.endswith(S)

if is_prefix and is_suffix:
    print(0)
elif is_prefix:
    print(1)
elif is_suffix:
    print(2)
else:
    print(3)