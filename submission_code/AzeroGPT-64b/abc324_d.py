from itertools import permutations
import math

def is_square(n):
    return int(math.sqrt(n)) ** 2 == n

def count_square_numbers(S):
    N = len(S)
    striN = '0' * (14 - N) + S
    ans = 0
    used = [False] * 14
    for perm in permutations(range(N)):
        num = 0
        for i, p in enumerate(perm):
            num += int(striN[p]) * 10 ** (N - i - 1)
        if is_square(num):
            ans += 1
    return ans

N = int(input())
S = input()
print(count_square_numbers(S))