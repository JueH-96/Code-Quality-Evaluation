import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S = data[2]
T = data[3]

is_prefix = S == T[:N]
is_suffix = S == T[-N:]

if is_prefix and is_suffix:
    print(0)
elif is_prefix:
    print(1)
elif is_suffix:
    print(2)
else:
    print(3)