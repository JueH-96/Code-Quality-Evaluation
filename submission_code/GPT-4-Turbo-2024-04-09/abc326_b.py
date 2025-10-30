def is_326_like(number):
    str_num = str(number)
    if len(str_num) != 3:
        return False
    hundreds = int(str_num[0])
    tens = int(str_num[1])
    ones = int(str_num[2])
    return hundreds * tens == ones

def find_next_326_like(N):
    while True:
        if is_326_like(N):
            return N
        N += 1

import sys
input = sys.stdin.read
N = int(input().strip())
result = find_next_326_like(N)
print(result)