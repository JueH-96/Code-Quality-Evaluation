import math

def find_smallest_326_like_number(N):
    for num in range(N, 920):
        hundreds = num // 100
        tens = (num // 10) % 10
        ones = num % 10
        if hundreds * tens == ones:
            return num
    return -1

N = int(input())
print(find_smallest_326_like_number(N))