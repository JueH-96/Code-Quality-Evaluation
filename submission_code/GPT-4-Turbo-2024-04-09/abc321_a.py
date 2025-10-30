def is_321_like_number(n):
    n_str = str(n)
    for i in range(len(n_str) - 1):
        if n_str[i] <= n_str[i + 1]:
            return "No"
    return "Yes"

import sys
input = sys.stdin.read
N = int(input().strip())
print(is_321_like_number(N))