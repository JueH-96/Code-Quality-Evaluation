# YOUR CODE HERE
import sys

def f(x, y):
    return (x + y) % 100000000

def solve(n, a):
    total = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            total += f(a[i], a[j])
    return total

input = sys.stdin.read
data = input().split()
n = int(data[0])
a = list(map(int, data[1:]))

print(solve(n, a))