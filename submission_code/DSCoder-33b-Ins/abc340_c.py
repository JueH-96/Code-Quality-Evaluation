# YOUR CODE HERE
N = int(input())

def solve(n):
    if n < 2:
        return 0
    else:
        return n + solve(n // 2) * 2

print(solve(N))