from sys import stdin

def solve(n):
    if n % 3 == 0:
        return "First"
    elif n % 3 == 1:
        return "Second"
    else:
        return "First"

t = int(stdin.readline())
for _ in range(t):
    n = int(stdin.readline())
    print(solve(n))