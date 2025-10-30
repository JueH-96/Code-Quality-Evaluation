# YOUR CODE HERE
n, m = map(int, input().split())
s = input()
t = input()

is_prefix = t[:n] == s
is_suffix = t[m-n:] == s

if is_prefix and is_suffix:
  print(0)
elif is_prefix:
  print(1)
elif is_suffix:
  print(2)
else:
  print(3)