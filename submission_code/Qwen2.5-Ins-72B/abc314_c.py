# YOUR CODE HERE
import sys

def right_circular_shift(s, indices):
    n = len(indices)
    if n == 0:
        return s
    last_char = s[indices[-1]]
    for i in range(n - 1, 0, -1):
        s[indices[i]] = s[indices[i - 1]]
    s[indices[0]] = last_char
    return s

def solve(n, m, s, c):
    s = list(s)
    for color in range(1, m + 1):
        indices = [i for i in range(n) if c[i] == color]
        s = right_circular_shift(s, indices)
    return ''.join(s)

input = sys.stdin.read
data = input().split()

n = int(data[0])
m = int(data[1])
s = data[2]
c = list(map(int, data[3:]))

result = solve(n, m, s, c)
print(result)