import sys

def solve():
    b = int(sys.stdin.readline())
    for a in range(1, 16):
        power = pow(a, a)
        if power == b:
            print(a)
            return
    print("-1")

if __name__ == '__main__':
    solve()