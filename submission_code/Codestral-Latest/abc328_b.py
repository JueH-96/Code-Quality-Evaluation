# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = list(map(int, data[1:]))

def is_repdigit(n):
    s = str(n)
    return all(c == s[0] for c in s)

count = 0
for i in range(1, N + 1):
    for j in range(1, D[i - 1] + 1):
        if is_repdigit(i) and is_repdigit(j):
            count += 1

print(count)