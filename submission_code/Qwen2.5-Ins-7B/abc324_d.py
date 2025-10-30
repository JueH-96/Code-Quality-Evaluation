# YOUR CODE HERE
from itertools import permutations

def is_square(n):
    return int(n**0.5)**2 == n

n = int(input())
s = input()
digits = sorted(s, reverse=True)
count = 0

for perm in permutations(digits):
    num = int(''.join(perm))
    if num > 10**66:
        break
    if is_square(num):
        count += 1

print(count)