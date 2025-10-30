def count_snake_numbers(L, R):
    count = 0
    for x in range(L, R + 1):
        s = str(x)
        top_digit = s[0]
        if all(top_digit > d for d in s[1:]):
            count += 1
    return count

import sys
input = sys.stdin.read
L, R = map(int, input().split())
print(count_snake_numbers(L, R))