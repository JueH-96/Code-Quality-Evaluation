import sys

def solve():
    a, b = map(int, sys.stdin.readline().split())
    if a == b:
        print("-1")
    else:
        culprit = 6 - a - b
        print(culprit)

if __name__ == '__main__':
    solve()