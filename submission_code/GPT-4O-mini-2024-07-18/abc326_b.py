def is_326_like(num):
    hundreds = num // 100
    tens = (num // 10) % 10
    ones = num % 10
    return hundreds * tens == ones

def smallest_326_like_number(N):
    for num in range(N, 1000):
        if is_326_like(num):
            return num

import sys
input = sys.stdin.read
N = int(input().strip())
print(smallest_326_like_number(N))