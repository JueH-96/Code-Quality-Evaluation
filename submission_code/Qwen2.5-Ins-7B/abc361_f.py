# YOUR CODE HERE
from math import isqrt, log

def count_special_numbers(N):
    count = 0
    for b in range(2, isqrt(N) + 1):
        a = 2
        while (a ** b) <= N:
            count += 1
            a += 1
    return count

N = int(input())
print(count_special_numbers(N))