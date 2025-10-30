# YOUR CODE HERE
import sys

def solve(n, q, r, x):
    r.sort()
    for i in x:
        count = 0
        for j in r:
            if j <= i:
                count += 1
                i -= j
            else:
                break
        print(count)

n, q = map(int, sys.stdin.readline().split())
r = list(map(int, sys.stdin.readline().split()))
x = [int(sys.stdin.readline()) for _ in range(q)]
solve(n, q, r, x)