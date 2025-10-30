# YOUR CODE HERE
def check_prefix_suffix(N, M, S, T):
    is_prefix = T.startswith(S)
    is_suffix = T.endswith(S)

    if is_prefix and is_suffix:
        return 0
    elif is_prefix:
        return 1
    elif is_suffix:
        return 2
    else:
        return 3

import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
S = data[1]
T = data[2]

result = check_prefix_suffix(N, M, S, T)
print(result)