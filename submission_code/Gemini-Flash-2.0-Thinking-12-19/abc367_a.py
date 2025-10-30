import sys

def solve():
    a, b, c = map(int, sys.stdin.readline().split())
    if b < c:
        if a < b or a >= c:
            print("Yes")
        else:
            print("No")
    else:
        if c <= a < b:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    solve()