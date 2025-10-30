# YOUR CODE HERE

import sys

def solve(n, a, b, c):
    pairs = set()
    for i in range(n):
        for x in range(1, c[i]//a[i]+1):
            y = (c[i] - a[i]*x) // b[i]
            if y > 0:
                pairs.add((x, y))
    return len(pairs)

def main():
    t = int(sys.stdin.readline())
    for _ in range(t):
        n = int(sys.stdin.readline())
        a, b, c = [], [], []
        for _ in range(n):
            ai, bi, ci = map(int, sys.stdin.readline().split())
            a.append(ai)
            b.append(bi)
            c.append(ci)
        print(solve(n, a, b, c))

if __name__ == '__main__':
    main()