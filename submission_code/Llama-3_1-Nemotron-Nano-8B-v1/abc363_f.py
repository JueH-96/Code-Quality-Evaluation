import math

n = int(input())

def is_valid_palindrome(x):
    s = str(x)
    return s == s[::-1] and '0' not in s

if n < 10:
    print(n)
    exit()

s = str(n)
if is_valid_palindrome(n):
    print(s)
    exit()

max_a = int(math.isqrt(n))
found = False

for a in range(1, max_a + 1):
    if n % a != 0:
        continue
    b = n // a
    a_str = str(a)
    b_str = str(b)
    if a_str == b_str[::-1] and '0' not in a_str and '0' not in b_str:
        candidate = f"{a_str}*{b_str}"
        if 1 <= len(candidate) <= 1000:
            print(candidate)
            exit()

for a in range(1, max_a + 1):
    a_squared = a * a
    if n % a_squared != 0:
        continue
    b = n // a_squared
    a_str = str(a)
    if not is_valid_palindrome(a):
        continue
    b_str = str(b)
    if '0' in b_str:
        continue
    candidate = f"{a_str}*{b_str}*{a_str}"
    if 1 <= len(candidate) <= 1000:
        print(candidate)
        exit()

print(-1)