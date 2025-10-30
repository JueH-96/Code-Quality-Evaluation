# YOUR CODE HERE
from itertools import permutations

def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]

    for p in permutations(s):
        valid = True
        for i in range(n - 1):
            diff = 0
            for j in range(m):
                if p[i][j] != p[i+1][j]:
                    diff += 1
            if diff != 1:
                valid = False
                break
        if valid:
            print("Yes")
            return
    print("No")

solve()