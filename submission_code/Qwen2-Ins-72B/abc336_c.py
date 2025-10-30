from math import log10

N = int(input())

def next_good(n):
    s = str(n)
    for i, c in enumerate(s):
        if int(c) % 2 == 1:
            if i == 0:
                return '2' + '0' * (len(s) - 1)
            else:
                return s[:i] + str(int(s[i]) + 1 if int(s[i]) < 8 else 2) + '0' * (len(s) - i - 1)
    return str(int(s) + 2 if int(s[-1]) < 8 else 2) + '0' * (len(s) - 1)

def good(n):
    return all(int(c) % 2 == 0 for c in str(n))

def solve(N):
    if N <= 5:
        return str(2 * N - 2)
    n = 5
    while True:
        n = next_good(n)
        N -= 1
        if N == 0:
            return n

print(solve(N))