import math

def has_zero(x):
    return '0' in str(x)

N = int(input())

# Check if N itself is a valid palindrome with no zeros
s = str(N)
if s == s[::-1] and not has_zero(N):
    print(s)
    exit()

# Check for a*b*a structure
max_a = int(math.isqrt(N))
for a in range(1, max_a + 1):
    if N % (a * a) != 0:
        continue
    b = N // (a * a)
    if not has_zero(a) and not has_zero(b):
        print(f"{a}*{b}*{a}")
        exit()

print(-1)