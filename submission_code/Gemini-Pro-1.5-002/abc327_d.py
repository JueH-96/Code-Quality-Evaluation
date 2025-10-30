# YOUR CODE HERE
import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))

    for i in range(1 << n):
        x = []
        for j in range(n):
            if (i >> j) & 1:
                x.append(1)
            else:
                x.append(0)
        
        valid = True
        for k in range(m):
            if x[a[k]-1] == x[b[k]-1]:
                valid = False
                break
        
        if valid:
            print("Yes")
            return
    
    print("No")

solve()