# YOUR CODE HERE
n, m = map(int, input().split())
s = input()
t = input()

is_prefix = s == t[:n]
is_suffix = s == t[m-n:]

if is_prefix and is_suffix:
    print(0)
elif is_prefix and not is_suffix:
    print(1)
elif not is_prefix and is_suffix:
    print(2)
else:
    print(3)