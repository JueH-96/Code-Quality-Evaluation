import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    dp0 = 0
    dp1 = -float('inf')
    for x in A:
        temp0 = max(dp0, dp1 + 2 * x)
        temp1 = max(dp1, dp0 + x)
        dp0, dp1 = temp0, temp1
    print(max(dp0, dp1))

if __name__ == "__main__":
    main()