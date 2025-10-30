import sys

def solve():
    n = int(sys.stdin.readline())
    count = 0
    while n > 0 and n % 2 == 0:
        n //= 2
        count += 1
    print(count)

if __name__ == '__main__':
    solve()