import math

def count_numbers(N):
    count = 0
    a = 2
    while a**2 <= N:
        b = 2
        while a**b <= N:
            count += 1
            b += 1
        a += 1
    return count

N = int(input())
print(count_numbers(N))