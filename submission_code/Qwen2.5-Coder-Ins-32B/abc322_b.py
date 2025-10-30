# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
S = input[2]
T = input[3]

is_prefix = (T[:N] == S)
is_suffix = (T[-N:] == S)

if is_prefix and is_suffix:
    print(0)
elif is_prefix:
    print(1)
elif is_suffix:
    print(2)
else:
    print(3)