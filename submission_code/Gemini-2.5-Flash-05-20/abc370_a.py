import sys

def solve():
    L, R = map(int, sys.stdin.readline().split())

    if L == 1 and R == 0:
        print("Yes")
    elif L == 0 and R == 1:
        print("No")
    elif L == 1 and R == 1:
        print("Invalid")
    elif L == 0 and R == 0:
        print("Invalid")

if __name__ == '__main__':
    solve()