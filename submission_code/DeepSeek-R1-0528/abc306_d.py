import sys

def main():
    input = sys.stdin.readline
    n = int(input())
    dp0 = 0
    dp1 = -10**18

    for _ in range(n):
        x, y = map(int, input().split())
        if x == 0:
            new0 = max(dp0, dp0 + y, dp1 + y)
            new1 = dp1
        else:
            new0 = dp0
            new1 = max(dp1, dp0 + y)
        dp0, dp1 = new0, new1

    print(max(dp0, dp1))

if __name__ == '__main__':
    main()