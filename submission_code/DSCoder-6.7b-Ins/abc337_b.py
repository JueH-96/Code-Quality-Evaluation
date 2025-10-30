# YOUR CODE HERE

import sys

def is_extended_abc(s):
    a_count = 0
    b_count = 0
    c_count = 0

    for char in s:
        if char == 'A':
            a_count += 1
        elif char == 'B':
            b_count += 1
        elif char == 'C':
            c_count += 1
        else:
            return False

    return a_count * b_count * c_count == len(s)

s = sys.stdin.readline().strip()

if is_extended_abc(s):
    print('Yes')
else:
    print('No')