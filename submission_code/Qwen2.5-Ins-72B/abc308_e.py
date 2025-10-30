# YOUR CODE HERE
def mex(a, b, c):
    values = {a, b, c}
    for i in range(3):
        if i not in values:
            return i

def solve(n, a, s):
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if s[i] == 'M' and s[j] == 'E' and s[k] == 'X':
                    count += mex(a[i], a[j], a[k])
    return count

import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
a = list(map(int, data[1:n+1]))
s = data[n+1]

print(solve(n, a, s))